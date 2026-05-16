import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.embeddings.hf_multimodal_embedder import EmbeddingConfig, HuggingFaceMultimodalEmbedder
from backend.app.knowledge.hunt_runtime import HuntKnowledgeService


def test_hunt_knowledge_service_loads_pack_and_resolves_aliases(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(
        json.dumps({"embedding_dimension": 8, "license": "CC BY-SA", "license_url": "https://example.invalid"}),
        encoding="utf-8",
    )
    entity = {
        "id": "weapon:dolch-96",
        "type": "weapon",
        "name": "Dolch 96",
        "aliases": ["Dolch", "Weapons/Dolch 96"],
        "description": "Semi-automatic pistol.",
        "source_url": "https://huntshowdown.wiki.gg/wiki/Weapons/Dolch_96",
        "image_paths": ["media/images/weapon/dolch.png"],
        "key_values": {"Damage": "97"},
    }
    chunk = {
        "id": "weapon:dolch-96:chunk:0",
        "entity_id": "weapon:dolch-96",
        "entity_type": "weapon",
        "title": "Dolch 96",
        "text": "The Dolch 96 is a semi-automatic pistol using special ammo.",
        "source_url": entity["source_url"],
    }
    (pack / "entities.jsonl").write_text(json.dumps(entity) + "\n", encoding="utf-8")
    (pack / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    (pack / "media_index.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "entity_id": "weapon:dolch-96",
                    "entity_type": "weapon",
                    "local_path": "media/images/weapon/dolch-skin.png",
                    "alt": "Weapon 3D Dolch 96 Red Dog.jpg",
                    "source_url": "https://example.invalid/Weapon_3D_Dolch_96_Red_Dog.jpg",
                    "title": "Dolch 96",
                },
                {
                    "entity_id": "weapon:dolch-96",
                    "entity_type": "weapon",
                    "local_path": "media/images/weapon/dolch-precision.png",
                    "alt": "Weapon Dolch 96 Precision.png",
                    "source_url": "https://example.invalid/Weapon_Dolch_96_Precision.png",
                    "title": "Dolch 96",
                    "content_type": "image/png",
                },
                {
                    "entity_id": "weapon:dolch-96",
                    "entity_type": "weapon",
                    "local_path": "media/images/weapon/dolch-ammo.png",
                    "alt": "Steel Ball Ammo",
                    "source_url": "https://example.invalid/Ammo_Steel_Ball.png",
                    "title": "Dolch 96",
                    "content_type": "image/png",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(pack / "embeddings.npy", np.ones((1, 8), dtype=np.float32))

    service = HuntKnowledgeService(pack, embedder=HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=8)))

    assert service.available is True
    assert service.lookup("was this a dolch?")[0].id == "weapon:dolch-96"
    assert service.resolve_equipment("Red Dog").display_name == "Dolch 96"
    assert service.resolve_equipment("Dolch 96 Red Dog").display_name == "Dolch 96"
    assert service.resolve_equipment("Dolch 96 Precision").display_name == "Dolch 96 Precision"
    assert [path for _, path in service.reference_image_records({"weapon"})] == ["media/images/weapon/dolch-precision.png"]
    hits = service.search("special ammo pistol", top_k=1)
    assert hits[0].entity_name == "Dolch 96"


def test_hunt_knowledge_service_applies_known_variant_aliases() -> None:
    service = HuntKnowledgeService(Path("data/packs/hunt-knowledge-pack"))
    if not service.available:
        return

    resolved = service.resolve_equipment("Mosin Obrez Drum", entity_types={"weapon"})

    assert resolved is not None
    assert resolved.display_name == "Mosin Obrez Extended"


def test_hunt_knowledge_service_applies_common_equipment_alias_overrides(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps({"embedding_dimension": 8}), encoding="utf-8")
    entities = [
        ("tool:first-aid-kit", "tool", "First Aid Kit"),
        ("tool:choke-bombs", "tool", "Choke Bombs"),
        ("tool:flare-pistol", "tool", "Flare Pistol"),
        ("consumable:regeneration-shot", "consumable", "Regeneration Shot"),
    ]
    entity_rows = [
        {
            "id": entity_id,
            "type": entity_type,
            "name": name,
            "aliases": [name],
            "description": name,
            "source_url": f"https://example.invalid/{entity_id}",
            "image_paths": [],
            "key_values": {},
        }
        for entity_id, entity_type, name in entities
    ]
    chunk_rows = [
        {"id": f"{row['id']}:chunk:0", "entity_id": row["id"], "entity_type": row["type"], "text": row["name"]}
        for row in entity_rows
    ]
    (pack / "entities.jsonl").write_text("\n".join(json.dumps(row) for row in entity_rows) + "\n", encoding="utf-8")
    (pack / "chunks.jsonl").write_text("\n".join(json.dumps(row) for row in chunk_rows) + "\n", encoding="utf-8")

    service = HuntKnowledgeService(pack)

    assert service.resolve_equipment("healing satchel", entity_types={"tool"}).entity.name == "First Aid Kit"
    assert service.resolve_equipment("Scarfskin Satchel", entity_types={"tool"}).entity.name == "First Aid Kit"
    assert service.resolve_equipment("chock bomb", entity_types={"tool"}).entity.name == "Choke Bombs"
    assert service.resolve_equipment("signal flair", entity_types={"tool"}).entity.name == "Flare Pistol"
    assert service.resolve_equipment("regen shot", entity_types={"consumable"}).entity.name == "Regeneration Shot"


def test_shared_skin_alias_prefers_base_weapon_but_keeps_variant_alias(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps({"embedding_dimension": 8}), encoding="utf-8")
    entities = [
        {
            "id": "weapon:auto-5",
            "type": "weapon",
            "name": "Auto-5",
            "aliases": ["Auto-5", "No Clemency"],
            "description": "Large slot shotgun.",
            "source_url": "https://huntshowdown.wiki.gg/wiki/Weapons/Auto-5",
            "image_paths": [],
            "key_values": {"Skins": "No Clemency"},
        },
        {
            "id": "weapon:auto-4-shorty",
            "type": "weapon",
            "name": "Auto-4 Shorty",
            "aliases": ["Auto-4 Shorty", "No Clemency", "No Clemency: Shorty"],
            "description": "Medium slot short shotgun.",
            "source_url": "https://huntshowdown.wiki.gg/wiki/Weapons/Auto-4_Shorty",
            "image_paths": [],
            "key_values": {"Skins": "No Clemency, No Clemency: Shorty"},
        },
    ]
    chunks = [
        {"id": f"{entity['id']}:chunk:0", "entity_id": entity["id"], "entity_type": "weapon", "text": entity["name"]}
        for entity in entities
    ]
    (pack / "entities.jsonl").write_text("\n".join(json.dumps(entity) for entity in entities) + "\n", encoding="utf-8")
    (pack / "chunks.jsonl").write_text("\n".join(json.dumps(chunk) for chunk in chunks) + "\n", encoding="utf-8")

    service = HuntKnowledgeService(pack)

    assert service.resolve_equipment("No Clemency", entity_types={"weapon"}).entity.name == "Auto-5"
    assert service.resolve_equipment("No Clemency: Shorty", entity_types={"weapon"}).entity.name == "Auto-4 Shorty"
    assert service.resolve_weapon_skin("No Clemency").display_name == "Auto-5 (No Clemency skin)"
    assert service.resolve_weapon_skin("No Clemency: Shorty").display_name == "Auto-4 Shorty (No Clemency: Shorty skin)"


def test_shared_skin_alias_prefers_base_weapon_for_any_variant(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps({"embedding_dimension": 8}), encoding="utf-8")
    entities = [
        {
            "id": "weapon:base",
            "type": "weapon",
            "name": "Example Rifle",
            "aliases": ["Example Rifle", "Shared Skin"],
            "description": "Base weapon.",
            "source_url": "https://huntshowdown.wiki.gg/wiki/Weapons/Example_Rifle",
            "image_paths": [],
            "key_values": {"Skins": "Shared Skin"},
        },
        {
            "id": "weapon:variant",
            "type": "weapon",
            "name": "Example Rifle Deadeye",
            "aliases": ["Example Rifle Deadeye", "Shared Skin", "Shared Skin: Deadeye"],
            "description": "Scoped variant.",
            "source_url": "https://huntshowdown.wiki.gg/wiki/Weapons/Example_Rifle/Deadeye",
            "image_paths": [],
            "key_values": {"Skins": "Shared Skin, Shared Skin: Deadeye"},
        },
    ]
    chunks = [
        {"id": f"{entity['id']}:chunk:0", "entity_id": entity["id"], "entity_type": "weapon", "text": entity["name"]}
        for entity in entities
    ]
    (pack / "entities.jsonl").write_text("\n".join(json.dumps(entity) for entity in entities) + "\n", encoding="utf-8")
    (pack / "chunks.jsonl").write_text("\n".join(json.dumps(chunk) for chunk in chunks) + "\n", encoding="utf-8")

    service = HuntKnowledgeService(pack)

    assert service.resolve_equipment("Shared Skin", entity_types={"weapon"}).entity.name == "Example Rifle"
    assert service.resolve_equipment("Shared Skin: Deadeye", entity_types={"weapon"}).entity.name == "Example Rifle Deadeye"
    assert service.resolve_weapon_skin("Shared Skin").display_name == "Example Rifle (Shared Skin skin)"
    assert service.resolve_weapon_skin("Shared Skin: Deadeye").display_name == "Example Rifle Deadeye (Shared Skin: Deadeye skin)"


def test_weapon_skin_map_resolves_skin_names_to_canonical_weapon(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps({"embedding_dimension": 8}), encoding="utf-8")
    entities = [
        {
            "id": "weapon:weapons-mosin-obrez",
            "type": "weapon",
            "name": "Mosin Obrez",
            "aliases": ["Mosin Obrez"],
            "description": "Shortened Mosin.",
            "source_url": "https://huntshowdown.wiki.gg/wiki/Weapons/Mosin_Obrez",
            "image_paths": [],
            "key_values": {"Skins": "Sinner's Jezail, Rougarou, The Fifth Tale, Iron Fury"},
        },
        {
            "id": "weapon:weapons-mosin-nagant",
            "type": "weapon",
            "name": "Mosin-Nagant",
            "aliases": ["Mosin-Nagant"],
            "description": "Full-length Mosin.",
            "source_url": "https://huntshowdown.wiki.gg/wiki/Weapons/Mosin-Nagant",
            "image_paths": [],
            "key_values": {"Skins": "Bear's Tooth"},
        },
    ]
    chunks = [
        {"id": f"{entity['id']}:chunk:0", "entity_id": entity["id"], "entity_type": "weapon", "text": entity["name"]}
        for entity in entities
    ]
    (pack / "entities.jsonl").write_text("\n".join(json.dumps(entity) for entity in entities) + "\n", encoding="utf-8")
    (pack / "chunks.jsonl").write_text("\n".join(json.dumps(chunk) for chunk in chunks) + "\n", encoding="utf-8")

    service = HuntKnowledgeService(pack)

    assert service.weapon_skin_map()["Rougarou"] == "Mosin Obrez (Rougarou skin)"
    assert service.resolve_weapon_skin("visible Rougarou-marked firearm").display_name == "Mosin Obrez (Rougarou skin)"
    assert service.resolve_equipment("visible Rougarou-marked firearm", entity_types={"weapon"}).display_name == "Mosin Obrez (Rougarou skin)"
    assert service.resolve_weapon_skin("Bear's Tooth").display_name == "Mosin-Nagant (Bear's Tooth skin)"
