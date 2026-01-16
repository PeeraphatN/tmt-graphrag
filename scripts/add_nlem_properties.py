"""
Add NLEM (National List of Essential Medicines) properties to GP nodes.
Category: 6.6 Drugs affecting bone metabolism (บัญชี ง)
"""
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri, auth=(user, password))

# NLEM data from https://ndi.fda.moph.go.th/drug_national_detail/index/14042
# Section 6.6 Drugs affecting bone metabolism - บัญชี ง

# TMTIDs that EXACTLY match NLEM criteria
exact_match_tmtids = [
    # Alendronate (alendronic acid) 70 mg tablet - EXACT
    '210257',  # alendronic acid 70 mg film-coated tablet
    '210261',  # alendronic acid 70 mg tablet
    
    # Calcitonin sterile solution (NOT nasal spray) - EXACT
    '211239',  # calcitonin 100 iu/1 mL solution for injection/infusion
    '659462',  # calcitonin 50 iu/1 mL solution for injection/infusion
    
    # Pamidronate sterile solution/powder - EXACT
    '217970',  # pamidronate disodium 15 mg/5 mL concentrate
    '217991',  # pamidronate disodium 30 mg/10 mL concentrate
    '972617',  # pamidronate disodium 30 mg powder and solvent
    
    # Zoledronic acid 4 mg/5 mL - EXACT
    '862368',  # zoledronic acid 4 mg/5 mL concentrate
    '220269',  # zoledronic acid 4 mg powder for solution
]

# TMTIDs for PoC (same drug family but different strength/form)
poc_tmtids = [
    # Alendronic acid - other strengths (PoC)
    '210235',  # 10 mg tablet
    '210242',  # 5 mg tablet
    '210274',  # 5 mg + calcitriol combination
    '210288',  # 70 mg + colecalciferol 5600 iu
    '210290',  # 70 mg + colecalciferol 70 mcg
    '740839',  # 150 mg tablet
    '851914',  # 70 mg + colecalciferol 2800 iu
    '1257231', # 70 mg effervescent
    
    # Calcitonin nasal spray (NOT in NLEM but same drug)
    '527811',  # 100 iu nasal spray
    '528081',  # 50 iu nasal spray
    '690977',  # 200 iu nasal spray
    
    # Zoledronic acid - other strengths
    '220282',  # 5 mg/100 mL
    '760224',  # 4 mg/100 mL
]

with driver.session() as session:
    # Add properties to EXACT matches
    print("Adding NLEM properties to EXACT matches...")
    for tmtid in exact_match_tmtids:
        result = session.run('''
            MATCH (n:GP {tmtid: $tmtid})
            SET n.nlem = true,
                n.nlem_category = "ง",
                n.nlem_section = "6.6",
                n.nlem_name = "Drugs affecting bone metabolism",
                n.nlem_match_type = "exact"
            RETURN n.fsn as fsn
        ''', tmtid=tmtid)
        record = result.single()
        if record:
            print(f"  ✅ {tmtid}: {record['fsn'][:60]}...")
        else:
            print(f"  ⚠️ {tmtid}: Not found")

    # Add properties to PoC matches
    print("\nAdding NLEM properties to PoC matches...")
    for tmtid in poc_tmtids:
        result = session.run('''
            MATCH (n:GP {tmtid: $tmtid})
            SET n.nlem = true,
                n.nlem_category = "ง",
                n.nlem_section = "6.6",
                n.nlem_name = "Drugs affecting bone metabolism",
                n.nlem_match_type = "poc"
            RETURN n.fsn as fsn
        ''', tmtid=tmtid)
        record = result.single()
        if record:
            print(f"  📋 {tmtid}: {record['fsn'][:60]}...")
        else:
            print(f"  ⚠️ {tmtid}: Not found")

driver.close()

print(f"\n=== Done! Added NLEM properties to {len(exact_match_tmtids)} exact + {len(poc_tmtids)} PoC nodes ===")
