import os
import json
import numpy as np
from dotenv import load_dotenv

# Környezeti változók betöltése
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)

NEUTRAL_ANALYSIS = {
    "sentiment": 0.0, 
    "confidence": 0.0, 
    "fundamental_score": 0.0,
    "geopolitical_risk": 0.5, 
    "indoklas": "Nincs Gemini API kulcs vagy hiba - semleges alapértelmezés.",
    "source": "fallback",
}

def get_live_sentiment(ticker: str, context: str) -> dict:
    """
    Geopolitikai és fundamentális elemzés a Gemini API-tól, a letöltött valós hírkontextus alapján.
    Ha nincs kulcs vagy hiba történik, semleges pontszámot ad vissza.
    """
    if not GEMINI_AVAILABLE:
        return dict(NEUTRAL_ANALYSIS)
        
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = (
            f"Pénzügyi elemzőként értékeld a {ticker} részvényt a MAI napon, az alábbi KÖZVETLEN, "
            f"legfrissebb hírek alapján:\n\n{context}\n\n"
            "Vedd figyelembe az itt olvasható (1) piaci hangulatot, (2) fundamentális utalásokat "
            "és (3) geopolitikai kockázatokat. NE támaszkodj a belső memóriádra, kizárólag a fenti hírekre építs!\n\n"
            "KIZÁRÓLAG ezt a JSON-t add vissza, semmi mást (markdown blokk nélkül):\n"
            '{"sentiment": <float -1..1>, "confidence": <float 0..1>, '
            '"fundamental_score": <float -1..1>, "geopolitical_risk": <float 0..1>, '
            '"indoklas": "<max 3 mondat magyarul, hivatkozva a hírekre>"}'
        )
        
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        out = json.loads(resp.text)
        out["source"] = "gemini (grounded with live news)"
        
        # Pontszámok biztonságos levágása az érvényes tartományba
        out["sentiment"] = float(np.clip(out.get("sentiment", 0), -1, 1))
        out["confidence"] = float(np.clip(out.get("confidence", 0), 0, 1))
        out["fundamental_score"] = float(np.clip(out.get("fundamental_score", 0), -1, 1))
        out["geopolitical_risk"] = float(np.clip(out.get("geopolitical_risk", 0.5), 0, 1))
        return out
        
    except Exception as exc:
        out = dict(NEUTRAL_ANALYSIS)
        out["indoklas"] = f"Gemini hiba ({type(exc).__name__}) - semleges pontszámok."
        return out
