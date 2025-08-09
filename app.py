#!/usr/bin/env python3
"""
app.py — Streamlit wrapper for the SEM pipeline

Usage:
  pip install -r requirements.txt
  streamlit run app.py

This uses SerpApi. You can paste your API key into the UI (recommended)
or keep the SERPAPI_API_KEY constant below (not recommended for security).
"""

import os
import time
import json
import math
import csv
import yaml
import requests
import concurrent.futures
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import pandas as pd
import streamlit as st

# -------------------------
# COPY OF YOUR ORIGINAL PIPELINE (kept intact, only namespaced into functions)
# -------------------------
# WARNING: Do not hardcode API keys in production. The UI allows overriding the key.
SERPAPI_API_KEY = ""  # optional default; leave empty and paste key into UI

# --- Utilities ---
def clean_str(s):
    return "" if s is None else str(s).strip()

def domain_token(url):
    if not url: return ""
    tok = url.replace("https://","").replace("http://","").replace("www.","").split("/")[0].split(".")[0]
    return tok.lower()

def fetch_site_text(url, timeout=6):
    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript","svg"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True).lower()
        return text
    except Exception:
        return ""

def extract_seed_terms(site_text, max_terms=8):
    words = [w for w in site_text.split() if len(w) >= 3]
    freq = {}
    for w in words:
        freq[w] = freq.get(w,0) + 1
    sorted_tokens = [w for w,c in sorted(freq.items(), key=lambda x:-x[1])]
    seeds = []
    for t in sorted_tokens[:max_terms*2]:
        if t not in seeds:
            seeds.append(t)
        if len(seeds) >= max_terms:
            break
    # bigrams
    bigrams = {}
    for i in range(len(words)-1):
        b = words[i] + " " + words[i+1]
        bigrams[b] = bigrams.get(b,0) + 1
    for b,c in sorted(bigrams.items(), key=lambda x:-x[1])[:max_terms]:
        if b not in seeds:
            seeds.append(b)
        if len(seeds) >= max_terms:
            break
    return seeds[:max_terms]

def serpapi_query(q, location, key, engine="google"):
    params = {"engine": engine, "q": q, "location": location, "api_key": key}
    try:
        search = GoogleSearch(params)
        return search.get_dict()
    except Exception:
        return {}

def extract_candidates(results):
    candidates = []
    if not results:
        return candidates
    # related_searches
    for rel in results.get("related_searches", []) or []:
        if isinstance(rel, dict):
            text = rel.get("query") or rel.get("title") or rel.get("text")
        else:
            text = rel
        if text:
            candidates.append({"keyword": clean_str(text), "source":"serpapi"})
    # questions / people also ask
    for q in results.get("questions", []) or results.get("related_questions", []) or []:
        text = q.get("question") if isinstance(q, dict) else q
        if text:
            candidates.append({"keyword": clean_str(text), "source":"serpapi"})
    # organic result titles
    for org in results.get("organic_results", []) or []:
        t = org.get("title") or org.get("snippet")
        if t:
            candidates.append({"keyword": clean_str(t), "source":"serpapi"})
    return candidates

def attempt_extract_metrics(results, candidate_text):
    if not results:
        return None, None, None, None
    for it in results.get("related_searches", []) or []:
        q = it.get("query") if isinstance(it, dict) else it
        if not q:
            continue
        if candidate_text.lower() in str(q).lower():
            sv = it.get("search_volume") or it.get("monthly_searches") or it.get("search_volume_monthly")
            low = it.get("top_of_page_bid_low") or it.get("low_top_of_page_bid_micros")
            high = it.get("top_of_page_bid_high") or it.get("high_top_of_page_bid_micros")
            comp = it.get("competition")
            if isinstance(low, (int,float)) and low > 1000:
                low = low / 1_000_000.0
            if isinstance(high, (int,float)) and high > 1000:
                high = high / 1_000_000.0
            return sv, low, high, comp
    si = results.get("search_information") or {}
    total = si.get("total_results")
    if total:
        return total, None, None, None
    return None, None, None, None

def estimate_volume(keyword, site_text):
    k = keyword.lower()
    words = k.split()
    score = 0
    if len(words) == 1: score += 60
    elif len(words) == 2: score += 40
    else: score += 20
    for tok in ["buy","price","best","review","near me","shop","order","service"]:
        if tok in k: score += 30
    if site_text and k in site_text: score += 40
    if score <= 20: return 50
    if score <= 60: return 300
    if score <= 110: return 1000
    if score <= 160: return 5000
    return 15000

def estimate_cpc(keyword):
    k = keyword.lower()
    base = 0.8
    for tok in ["software","saas","platform","service","buy","price","order","quote"]:
        if tok in k: base += 0.8
    words = len(k.split())
    if words == 1: base += 1.0
    elif words == 2: base += 0.5
    return round(max(0.05, base), 2)

def dedupe_list(items):
    seen = set()
    out = []
    for it in items:
        kw = clean_str(it.get("keyword") if isinstance(it, dict) else it)
        if not kw: continue
        key = kw.lower()
        if key in seen: continue
        seen.add(key)
        entry = {"keyword": kw}
        for fld in ("search_volume","cpc","top_low","top_high","competition","source"):
            if isinstance(it, dict) and it.get(fld) is not None:
                entry[fld] = it.get(fld)
        out.append(entry)
    return out

def finalize_candidates(candidates, site_text, min_vol, min_cpc, max_cpc, allow_fallback):
    final = []
    for it in candidates:
        kw = it["keyword"]
        sv = it.get("search_volume")
        cpc = it.get("cpc")
        low = it.get("top_low")
        high = it.get("top_high")
        src = it.get("source","serpapi")
        if allow_fallback:
            if sv is None:
                sv = estimate_volume(kw, site_text)
            if cpc is None:
                if low and high:
                    try:
                        cpc = round((float(low)+float(high))/2.0,2)
                    except:
                        cpc = estimate_cpc(kw)
                else:
                    cpc = estimate_cpc(kw)
            if low is None or high is None:
                low = round(cpc*0.8,2)
                high = round(cpc*1.2,2)
            src = src if src=="serpapi" else "estimated"
        else:
            if sv is None or cpc is None:
                continue
            if low is None or high is None:
                low = round(cpc*0.8,2)
                high = round(cpc*1.2,2)
        try:
            sv_n = int(sv)
            cpc_n = float(cpc)
        except:
            continue
        if sv_n < min_vol: continue
        if cpc_n < min_cpc or cpc_n > max_cpc: continue
        final.append({
            "keyword": kw,
            "search_volume": sv_n,
            "cpc": round(cpc_n,2),
            "cpc_low": round(float(low),2),
            "cpc_high": round(float(high),2),
            "competition": it.get("competition","unknown"),
            "source": src
        })
    return final

def label_intent(keyword, brand_site, comp_site, locations):
    k = keyword.lower()
    brand_tok = domain_token(brand_site)
    comp_tok = domain_token(comp_site)
    if brand_tok and brand_tok in k: return "Brand Terms"
    if comp_tok and comp_tok in k: return "Competitor Terms"
    if any(loc.lower() in k for loc in locations): return "Location-based Queries"
    if k.startswith(("how ","what ","why ","best ")): return "Long-Tail Informational Queries"
    return "Category Terms"

def suggest_match_types(keyword, intent):
    k = keyword.lower()
    if intent == "Brand Terms": return "Exact, Phrase"
    if intent == "Competitor Terms": return "Phrase, Exact"
    if intent == "Long-Tail Informational Queries": return "Phrase"
    if len(k.split()) == 1: return "Broad Match Modifier"
    return "Phrase, Broad Match Modifier"

def derive_pmax_from_groups(grouped):
    themes = {
        "Product Category Themes": [],
        "Use-case Based Themes": [],
        "Demographic Themes": [],
        "Seasonal/Event-Based Themes": []
    }
    if "Category Terms" in grouped:
        top = sorted(grouped["Category Terms"], key=lambda x: x["search_volume"], reverse=True)[:8]
        themes["Product Category Themes"] = [t["keyword"] for t in top]
    if "Long-Tail Informational Queries" in grouped:
        top = sorted(grouped["Long-Tail Informational Queries"], key=lambda x: x["search_volume"], reverse=True)[:8]
        themes["Use-case Based Themes"] = [t["keyword"] for t in top]
    demos = []
    for grp in grouped.values():
        for k in grp:
            if " for " in k["keyword"].lower():
                demos.append(k["keyword"])
    themes["Demographic Themes"] = demos[:8]
    seasonal_tokens = ["back to school","black friday","cyber monday","holiday","christmas","summer","winter","new year"]
    seas = []
    for grp in grouped.values():
        for k in grp:
            if any(tok in k["keyword"].lower() for tok in seasonal_tokens):
                seas.append(k["keyword"])
    themes["Seasonal/Event-Based Themes"] = seas[:8]
    return themes

def compute_shopping_bids(finalized, shopping_budget, conv_rate=0.02):
    rows = []
    for k in finalized:
        mid = k["cpc"]
        target_cpa_proxy = mid
        target_cpc = round(target_cpa_proxy * conv_rate, 4)
        recommended = round(max(target_cpc, mid * 0.8), 4)
        rows.append({
            "product": k["keyword"],
            "top_of_page_bid_low": k["cpc_low"],
            "top_of_page_bid_high": k["cpc_high"],
            "competition": k["competition"],
            "target_cpa_proxy": round(target_cpa_proxy,4),
            "target_cpc": target_cpc,
            "recommended_cpc": recommended,
            "search_volume": k["search_volume"]
        })
    return rows

def export_deliverable1(grouped):
    rows = []
    for group, kws in grouped.items():
        for k in kws:
            rows.append({
                "Final list of filtered keywords": k["keyword"],
                "Suggested match types (Broad Match Modifier, Phrase, Exact)": suggest_match_types(k["keyword"], group),
                "Suggested CPC range based on bid benchmarks": f"{k['cpc_low']} - {k['cpc_high']}"
            })
    df = pd.DataFrame(rows)
    return df

def export_deliverable2(themes):
    row = {
        "Product Category Themes": "; ".join(themes.get("Product Category Themes", [])),
        "Use-case Based Themes": "; ".join(themes.get("Use-case Based Themes", [])),
        "Demographic Themes": "; ".join(themes.get("Demographic Themes", [])),
        "Seasonal/Event-Based Themes": "; ".join(themes.get("Seasonal/Event-Based Themes", []))
    }
    df = pd.DataFrame([row])
    return df

def export_deliverable3(shopping_rows):
    df = pd.DataFrame([{
        "Product Name": r["product"],
        "Top of Page Bid Low": r["top_of_page_bid_low"],
        "Top of Page Bid High": r["top_of_page_bid_high"],
        "Competition": r["competition"],
        "Target CPC": r["target_cpc"],
        "Recommended CPC": r["recommended_cpc"],
        "Search Volume": r["search_volume"]
    } for r in shopping_rows])
    return df

# -------------------------
# Streamlit UI + runner
# -------------------------
st.set_page_config(page_title="SEM Keyword Pipeline", layout="wide")
st.title("SEM Keyword Pipeline")

st.markdown("""
Upload a YAML or enter inputs in the sidebar, paste your SerpApi key, and click **Run pipeline**.
The app shows three deliverables and gives download buttons for CSV/JSON.
""")

with st.sidebar:
    st.header("Inputs")
    ui_api_key = st.text_input("SerpApi Key (paste here to override)", type="password", value="")
    ui_brand_site = st.text_input("Brand website (full URL)", value="")
    ui_comp_site = st.text_input("Competitor website (full URL, optional)", value="")
    ui_locations = st.text_input("Service locations (comma-separated)", value="")
    ui_shopping_budget = st.number_input("Shopping ads budget", value=500.0)
    ui_search_budget = st.number_input("Search ads budget", value=1000.0)
    ui_pmax_budget = st.number_input("PMax ads budget", value=800.0)
    ui_min_search_volume = st.number_input("Min search volume filter", value=500)
    ui_min_cpc = st.number_input("Min CPC filter", value=0.5, format="%.2f")
    ui_max_cpc = st.number_input("Max CPC filter", value=5.0, format="%.2f")
    ui_allow_fallback = st.checkbox("Allow fallback estimation for missing metrics", value=True)
    ui_top_n_per_seed = st.number_input("Top N per seed (controls runtime)", value=30, min_value=5, max_value=200)
    uploaded_yaml = st.file_uploader("Or upload config.yaml (optional)", type=["yaml","yml"])
    run_pipeline = st.button("Run pipeline")

# If YAML uploaded, parse and override UI entries
config_overrides = {}
if uploaded_yaml is not None:
    try:
        cfg = yaml.safe_load(uploaded_yaml.read())
        config_overrides = cfg or {}
        st.sidebar.success("YAML loaded — fields overridden where provided.")
    except Exception as e:
        st.sidebar.error(f"Error reading YAML: {e}")

# Prepare final inputs (UI value wins unless YAML provided)
api_key = (config_overrides.get("f46a322763ea2ee11b3a0c77bb68e4beae0b78a2b2293bff44b4b5595e503d92") or ui_api_key or SERPAPI_API_KEY).strip()
brand_site = config_overrides.get("brand_website") or ui_brand_site
comp_site = config_overrides.get("competitor_website") or ui_comp_site
locations = config_overrides.get("service_locations") or [l.strip() for l in ui_locations.split(",") if l.strip()]
shopping_budget = float(config_overrides.get("shopping_ads_budget", ui_shopping_budget))
search_budget = float(config_overrides.get("search_ads_budget", ui_search_budget))
pmax_budget = float(config_overrides.get("pmax_ads_budget", ui_pmax_budget))
min_search_volume = int(config_overrides.get("min_search_volume", ui_min_search_volume))
min_cpc = float(config_overrides.get("min_cpc", ui_min_cpc))
max_cpc = float(config_overrides.get("max_cpc", ui_max_cpc))
allow_fallback = bool(config_overrides.get("allow_fallback", ui_allow_fallback))
top_n_per_seed = int(config_overrides.get("top_n_per_seed", ui_top_n_per_seed))

# Output dir
outdir = "output"
os.makedirs(outdir, exist_ok=True)

if run_pipeline:
    if not api_key:
        st.error("SerpApi key required. Please paste it in the sidebar (it will override any key in the script).")
    elif not brand_site or not locations:
        st.error("Please provide a brand website and at least one location.")
    else:
        t0 = time.time()
        st.info("Running pipeline — please wait. Progress bar shows seed query progress.")
        progress = st.progress(0)

        # fetch site content and derive seeds
        site_text = ""
        if brand_site:
            site_text += " " + fetch_site_text(brand_site)
        if comp_site:
            site_text += " " + fetch_site_text(comp_site)
        site_text = site_text.lower()

        brand_tok = domain_token(brand_site)
        comp_tok = domain_token(comp_site)
        seeds = []
        if brand_tok: seeds.append(brand_tok)
        if comp_tok: seeds.append(comp_tok)
        seeds += extract_seed_terms(site_text, max_terms=6)
        # build combined seeds with locations
        combined = []
        for s in seeds:
            combined.append(s)
            for loc in locations:
                combined.append(f"{s} {loc}")
        combined = list(dict.fromkeys([c for c in combined if c]))
        seeds_to_query = combined[: min(12, len(combined))]
        st.write(f"Seeds to query (capped): {seeds_to_query}")

        candidate_items = []
        # parallel queries
        tasks = len(seeds_to_query)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            futures = { ex.submit(serpapi_query, s, locations[0] if locations else "United States", api_key, "google"): s for s in seeds_to_query }
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                seed = futures[fut]
                res = fut.result()
                cands = extract_candidates(res)
                limited = cands[: max(10, top_n_per_seed//3)]
                for cand in limited:
                    # try small metric extraction re-query
                    res2 = serpapi_query(cand["keyword"], locations[0] if locations else "United States", api_key, "google")
                    sv, low, high, comp = attempt_extract_metrics(res2, cand["keyword"])
                    item = {
                        "keyword": cand["keyword"],
                        "search_volume": sv,
                        "top_low": low,
                        "top_high": high,
                        "cpc": None,
                        "competition": comp,
                        "source": "serpapi"
                    }
                    if item["top_low"] and item["top_high"]:
                        try:
                            item["cpc"] = round((float(item["top_low"]) + float(item["top_high"])) / 2.0, 2)
                        except:
                            item["cpc"] = None
                    candidate_items.append(item)
                completed += 1
                progress.progress(int(completed / tasks * 100))

        # fallback to google suggest if empty
        if not candidate_items:
            st.warning("No candidates from SerpApi — falling back to Google Suggest for seeds.")
            for s in seeds_to_query:
                try:
                    url = "https://suggestqueries.google.com/complete/search?client=firefox&q=" + quote_plus(s)
                    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=4)
                    arr = r.json()
                    suggestions = arr[1] if isinstance(arr, list) and len(arr) > 1 else []
                    for sug in suggestions[:top_n_per_seed]:
                        candidate_items.append({"keyword": clean_str(sug), "search_volume": None, "cpc": None, "top_low": None, "top_high": None, "competition":None, "source":"autocomplete"})
                except Exception:
                    pass

        candidate_items = dedupe_list(candidate_items)
        cap = max(200, top_n_per_seed * len(seeds_to_query))
        candidate_items = candidate_items[:cap]
        st.write(f"Candidate items collected: {len(candidate_items)}")

        # finalize
        finalized = finalize_candidates(candidate_items, site_text, min_search_volume, min_cpc, max_cpc, allow_fallback)
        st.write(f"Finalized keywords after filters: {len(finalized)}")
        if not finalized:
            st.error("No keywords detected after applying filters. Try allowing fallback or relaxing thresholds.")
        else:
            # label and group
            for k in finalized:
                k["intent"] = label_intent(k["keyword"], brand_site, comp_site, locations)
                k["match_types"] = suggest_match_types(k["keyword"], k["intent"])
            grouped = {}
            for k in finalized:
                grouped.setdefault(k["intent"], []).append(k)

            # deliverable 1
            df1 = export_deliverable1(grouped)
            st.subheader("Deliverable 1 — Filtered Keyword List (Search Campaign)")
            st.dataframe(df1)
            st.download_button("Download Deliverable 1 CSV", df1.to_csv(index=False).encode("utf-8"), "deliverable1_search_keywords.csv", "text/csv")
            df1.to_csv(os.path.join(outdir, "deliverable1_search_keywords.csv"), index=False)

            # deliverable 2 (PMax themes)
            themes = derive_pmax_from_groups(grouped)
            df2 = export_deliverable2(themes)
            st.subheader("Deliverable 2 — Performance Max Themes")
            st.dataframe(df2)
            st.download_button("Download Deliverable 2 CSV", df2.to_csv(index=False).encode("utf-8"), "deliverable2_pmax_themes.csv", "text/csv")
            df2.to_csv(os.path.join(outdir, "deliverable2_pmax_themes.csv"), index=False)

            # deliverable 3 (shopping CPC)
            shopping_rows = compute_shopping_bids(finalized, shopping_budget, conv_rate=0.02)
            df3 = export_deliverable3(shopping_rows)
            st.subheader("Deliverable 3 — Suggested Shopping CPCs")
            st.dataframe(df3)
            st.download_button("Download Deliverable 3 CSV", df3.to_csv(index=False).encode("utf-8"), "deliverable3_shopping_cpc.csv", "text/csv")
            df3.to_csv(os.path.join(outdir, "deliverable3_shopping_cpc.csv"), index=False)

            # JSON export for provenance
            json_out = os.path.join(outdir, f"keywords_grouped_{int(time.time())}.json")
            with open(json_out, "w", encoding="utf-8") as jf:
                json.dump({"groups": grouped, "pmax_themes": themes, "shopping": shopping_rows}, jf, ensure_ascii=False, indent=2)
            st.success(f"Pipeline completed in {int(time.time()-t0)}s — outputs saved to `{outdir}/`.")
            st.markdown(f"JSON summary: `{json_out}`")

# END OF app.py
