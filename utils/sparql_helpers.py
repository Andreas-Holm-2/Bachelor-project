from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import time


def run_query(query: str) -> list[dict]:
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "MyResearchBot")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    sparql.setMethod("POST")
    sparql.setRequestMethod("postdirectly")
    return sparql.query().convert()["results"]["bindings"]



def fetch_property_batched(qids: list[str], prop: str, col_name: str) -> pd.DataFrame:
    CHUNK = 30000
    rows = []
    for i in range(0, len(qids), CHUNK):
        chunk = qids[i:i+CHUNK]
        values_block = " ".join(f"wd:{q}" for q in chunk)
        print(f"  [{col_name}] chunk {i//CHUNK + 1}/{(len(qids)-1)//CHUNK + 1}")
        try:
            results = run_query(f"""
                SELECT ?item ?value ?valueLabel WHERE {{
                  VALUES ?item {{ {values_block} }}
                  OPTIONAL {{ ?item wdt:{prop} ?value . }}
                  SERVICE wikibase:label {{
                    bd:serviceParam wikibase:language "en,da".
                  }}
                }}
            """)
            for r in results:
                qid = r["item"]["value"].split("/")[-1]
                val = r.get("valueLabel", r.get("value", {})).get("value")
                if val:
                    rows.append({"item": qid, col_name: val})
        except Exception as e:
            print(f"  Error: {e} — skipping chunk")
        time.sleep(2)

    if not rows:
        return pd.DataFrame(columns=["item", col_name])

    return (
        pd.DataFrame(rows)
        .groupby("item")[col_name]
        .apply(lambda x: list(x.unique()))
        .reset_index()
    )


def fetch_date_with_precision(qids: list[str], prop: str, col_name: str) -> pd.DataFrame:
    CHUNK = 30000
    rows = []
    for i in range(0, len(qids), CHUNK):
        chunk = qids[i:i+CHUNK]
        values_block = " ".join(f"wd:{q}" for q in chunk)
        print(f"  [{col_name}] chunk {i//CHUNK + 1}/{(len(qids)-1)//CHUNK + 1}")
        try:
            results = run_query(f"""
                SELECT ?item ?date ?precision WHERE {{
                  VALUES ?item {{ {values_block} }}
                  OPTIONAL {{
                    ?item p:{prop} ?stmt .
                    ?stmt psv:{prop} ?dateValue .
                    ?dateValue wikibase:timeValue ?date .
                    ?dateValue wikibase:timePrecision ?precision .
                  }}
                }}
            """)
            for r in results:
                qid = r["item"]["value"].split("/")[-1]
                date = r.get("date", {}).get("value")
                precision = r.get("precision", {}).get("value")
                if date:
                    rows.append({
                        "item": qid,
                        col_name: date,
                        f"{col_name}_precision": int(precision) if precision else None,
                    })
        except Exception as e:
            print(f"  Error: {e} — skipping chunk")
        time.sleep(2)

    if not rows:
        return pd.DataFrame(columns=["item", col_name, f"{col_name}_precision"])

    return pd.DataFrame(rows).drop_duplicates("item")


def fetch_property_with_qualifiers(
    qids: list[str], prop: str, col_name: str, qualifiers: dict[str, str]
) -> pd.DataFrame:

    CHUNK = 30000
    rows = []

    qualifier_selects = " ".join(f"?{name} ?{name}Label" for name in qualifiers)
    qualifier_optionals = "\n".join(
        f"          OPTIONAL {{ ?stmt pq:{pid} ?{name} . }}"
        for name, pid in qualifiers.items()
    )

    for i in range(0, len(qids), CHUNK):
        chunk = qids[i:i+CHUNK]
        values_block = " ".join(f"wd:{q}" for q in chunk)
        print(f"  [{col_name}] chunk {i//CHUNK + 1}/{(len(qids)-1)//CHUNK + 1}")
        try:
            results = run_query(f"""
                SELECT ?item ?value ?valueLabel {qualifier_selects} WHERE {{
                  VALUES ?item {{ {values_block} }}
                  OPTIONAL {{
                    ?item p:{prop} ?stmt .
                    ?stmt ps:{prop} ?value .
{qualifier_optionals}
                  }}
                  SERVICE wikibase:label {{
                    bd:serviceParam wikibase:language "en,da,mul,[AUTO_LANGUAGE]".
                  }}
                }}
            """)
            for r in results:
                qid = r["item"]["value"].split("/")[-1]
                label = r.get("valueLabel", r.get("value", {})).get("value")
                if not label:
                    continue
                entry = {"value": label}
                for name in qualifiers:
                    raw   = r.get(name, {}).get("value")
                    resolved = r.get(f"{name}Label", {}).get("value")
                    entry[name] = resolved if resolved else raw
                rows.append({"item": qid, col_name: entry})
        except Exception as e:
            print(f"  Error: {e} — skipping chunk")
        time.sleep(2)

    if not rows:
        return pd.DataFrame(columns=["item", col_name])

    return (
        pd.DataFrame(rows)
        .groupby("item")[col_name]
        .apply(list)
        .reset_index()
    )
