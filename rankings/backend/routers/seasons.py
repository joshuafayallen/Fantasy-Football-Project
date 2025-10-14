from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path
import json


router = APIRouter(prefix="/seasons", tags=["seasons"])

DATA_FILE = Path("seasons_rankings.json")


def load_precomputed():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            return json.load(f)


@router.get("/")
def list_seasons():
    results = load_precomputed()
    return {"available_seasons": list(results.keys())}


@router.get("/{season}")
def get_seasson(season: int):
    data = load_precomputed()
    key = f"{season} Season"
    if key not in data:
        available = sorted(data.keys())
        return JSONResponse(
            status_code=404,
            content={
                "error": f"{season} not found\n",
                "Available Seasons": available,
            },
        )
    return data[key]
