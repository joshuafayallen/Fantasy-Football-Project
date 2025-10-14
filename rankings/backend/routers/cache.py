from fastapi import APIRouter
from pathlib import Path

router = APIRouter(prefix="/cache", tags=["cache"])

DATA_FILE = Path("seasons_rankings.json")


@router.delete("/clear")
def clear_cache():
    if DATA_FILE.exists():
        DATA_FILE.unlink()
        return {"message": "Cache Cleared"}
    return {"message": "No cache file found"}
