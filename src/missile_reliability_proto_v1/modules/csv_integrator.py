import os
import pandas as pd
import numpy as np

class CSVIntegrator:
    def __init__(self):
        self.item_map = self._build_item_mapping()

    def _build_item_mapping(self):
        # Hardcoded reference (Shared with PDF Converter logic)
        ref_items = {
            "1": "열전지스퀴브저항", "2": "사출모터스퀴브저항",
            "3": "22P - PGND", "4": "PGND - 22P", "5": "22N - PGND", "6": "PGND - 22N", "7": "22P - 22N", "8": "22N - 22P",
            "9": "자전기동시간", "10": "자전안정화주파수",
            "11": "ASLAP", "12": "ASLAY", "13": "ALOSRP", "14": "ALOSRY", "15": "AVIFSC", "16": "AFROLL", "17": "AMRSIN",
            "18": "+5V", "19": "+15V", "20": "-15V",
            "21": "SLAP", "22": "SLAY", "23": "SLAP", "24": "SLAY", "25": "SLAP", "26": "SLAY",
            "27": "냉각유지시간"
        }
        
        mapping = {}
        for id, name in ref_items.items():
            # Standardize Key: No spaces, No dashes, Upper
            clean_name = name.replace(" ", "").replace("-", "").upper()
            if clean_name not in mapping:
                mapping[clean_name] = []
            mapping[clean_name].append(id)
            
        # Manual variations
        variations = {
            "자전안정화": ["10"], "자전안정화주파수(f0)": ["10"],
            "보조채널조종(ACPA1)": ["15"], "5'측정(ACPA)": ["15"], 
            "시선각(SLAY)": ["22", "24", "26"],
            "22PPGND": ["3"], "PGND22P": ["4"], "22NPGND": ["5"], "PGND22N": ["6"],
            "22P22N": ["7"], "22N22P": ["8"],
            "냉각유지": ["27"], "냉각유지시간": ["27"]
        }
        for k, v in variations.items():
            clean_k = k.replace(" ", "").replace("-", "").upper()
            if clean_k not in mapping: mapping[clean_k] = v
            else: mapping[clean_k].extend(v) 
             
        # Specific fix for Squib ambiguity from PDF logic
        mapping['열전지스퀴브'] = ['1']
        
        return mapping

    def _correct_item_name(self, name: str) -> str:
        if not isinstance(name, str): return str(name)
        name_clean = name.replace(" ", "").replace("(", "").replace(")", "")
        
        # Aggressive correction for common OCR errors
        if "스귀스" in name_clean or "스키브" in name_clean:
            name_clean = name_clean.replace("스귀스", "스퀴브").replace("스키브", "스퀴브")
        if "저향" in name_clean:
            name_clean = name_clean.replace("저향", "저항")
            
        corrections = {
            "자전안정화": "자전 안정화 주파수(f0)",
            "보조채널조증ACPA7": "보조채널조종(ACPA1)", "보조채널조A0PA2": "보조채널조종(ACPA2)",
            "보조채널조증AOPA": "보조채널조종(ACPA2)", "보조채널조증A0PA3": "보조채널조종(ACPA3)",
            "보조채닐조증": "보조채널조종(ACPA3)", "보조채널조증": "보조채널조종(ACPA1)",
            "시선PSAY": "시선각(SLAY)", "5혹정스PA": "5' 측정(ACPA)",
            "보조지널획의스PA": "보조채널 확인(ACPA)", "보조지널": "보조채널 확인(ACPA)",
            "열전지스퀴브": "열전지스퀴브저항"
        }
        
        for key, val in corrections.items():
            if key in name_clean: return val
        return name_clean

    def integrate_folder(self, folder_path: str) -> pd.DataFrame:
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'sample' not in f and 'converted' not in f]
        files.sort()
        
        if not files: return pd.DataFrame()

        # Init Row Structure
        row = {
            '일련번호': '',
            '품번': '81040050', 
            '시험일자': '', 
            '운용월': 0,
            'Code': '',
            'LOT 번호': '',
            '합격여부': 'Unknown',
            '불합격 항목': ''
        }
        for i in range(1, 28): row[str(i)] = np.nan
        
        name_counts = {}
        failed_items_list = []
        
        sn_found = False
        date_found = False
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                try:
                    df_raw = pd.read_csv(file_path, header=None, names=range(30), encoding='utf-8', engine='python')
                except UnicodeDecodeError:
                    df_raw = pd.read_csv(file_path, header=None, names=range(30), encoding='cp949', engine='python')
                
                # Metadata extraction (simplified)
                if not sn_found and len(df_raw) > 1:
                    row1 = df_raw.iloc[1].astype(str).values
                    for cell in row1:
                        if cell.startswith('SG'):
                            row['일련번호'] = cell.split(',')[0]
                            sn_found = True
                            break
                            
                if not date_found and len(df_raw) > 6:
                    for r_idx in range(4, min(10, len(df_raw))):
                        row_vals = df_raw.iloc[r_idx].astype(str).values
                        for cell in row_vals:
                            if '20' in cell and '-' in cell and ':' in cell:
                                try:
                                    row['시험일자'] = cell.split()[0].replace('-', '.')
                                    date_found = True
                                except: pass
                                break
                        if date_found: break

                # Data loop
                for idx, r in df_raw.iterrows():
                    if len(r) < 11: continue
                    
                    # Dynamically detect Value/Result column
                    # Candidates for Result (Pass/Fail): Index 10 or 11
                    res_11 = str(r[11]) if len(r) > 11 else ""
                    res_10 = str(r[10])
                    
                    val_idx = -1
                    result_text = ""
                    
                    # Check known Pass/Fail markers
                    pass_markers = ['합격', 'PASS', 'P', 'OK', '정상', '정상수신']
                    fail_markers = ['불합격', 'FAIL', 'F', 'NG', '부적합']
                    all_markers = pass_markers + fail_markers
                    
                    if any(m in res_11.upper() for m in all_markers):
                        val_idx = 10
                        result_text = res_11
                    elif any(m in res_10.upper() for m in all_markers):
                        val_idx = 9
                        result_text = res_10
                    else:
                        # Fallback: Try decoding float at 10, then 9
                        try:
                            float(str(r[10]).replace(',', ''))
                            val_idx = 10
                            result_text = res_11
                        except:
                            try:
                                float(str(r[9]).replace(',', ''))
                                val_idx = 9
                                result_text = res_10
                            except:
                                continue # Cannot find value
                    
                    if val_idx == -1: continue
                    
                    value = r[val_idx]
                    
                    try:
                        clean_val = str(value).replace(',', '')
                        float(clean_val)
                    except:
                        continue 
                        
                    name_candidates = [str(r[2]), str(r[3]) if len(r) > 3 else ""]
                    
                    for name_raw in name_candidates:
                        if pd.isna(name_raw) or name_raw == 'nan' or not name_raw.strip(): continue
                        
                        # Pre-clean
                        if "+5V" in name_raw: name_raw = "+5V" 
                        
                        clean_name = self._correct_item_name(name_raw).replace(" ", "").replace("-", "").upper()
                        clean_name = clean_name.replace("·", "").replace(".", "")

                        # DEBUG: Only print interesting items to reduce spam if needed, or keep all
                        # print(f"DEBUG: Checking '{clean_name}' (Raw: '{name_raw}')")

                        item_id = None
                        matched_name = ""

                        # 1. Prioritize ASLAP/ASLAY logic
                        if "ASLAP" in clean_name: item_id = "11"
                        elif "ASLAY" in clean_name: item_id = "12"
                        elif "ALOSRP" in clean_name: item_id = "13"
                        elif "ALOSRY" in clean_name: item_id = "14"
                        
                        # 2. Filename-based SLAP/SLAY
                        elif "SLAP" in clean_name or "SLAY" in clean_name:
                            if "1500" in file:
                                if "SLAP" in clean_name: item_id = "21"
                                else: item_id = "22"
                            elif "1600" in file:
                                if "SLAP" in clean_name: item_id = "23"
                                else: item_id = "24"
                            elif "1700" in file:
                                if "SLAP" in clean_name: item_id = "25"
                                else: item_id = "26"
                        
                        # 3. Standard Map
                        else:
                            candidate_ids = self.item_map.get(clean_name)
                            if not candidate_ids:
                                 # Fuzzy
                                 fuzzy_candidates = []
                                 for k, v in self.item_map.items():
                                    if k in clean_name or clean_name in k: 
                                        fuzzy_candidates.extend(v)
                                 if fuzzy_candidates:
                                    candidate_ids = sorted(list(set(fuzzy_candidates)), key=lambda x: int(x))
                            
                            if candidate_ids:
                                if len(candidate_ids) == 1:
                                    item_id = candidate_ids[0]
                                else:
                                    # Sequential
                                    count_key = clean_name
                                    c_idx = name_counts.get(count_key, 0)
                                    if c_idx < len(candidate_ids):
                                        item_id = candidate_ids[c_idx]
                                        name_counts[count_key] = c_idx + 1
                                    else:
                                        item_id = candidate_ids[-1]
                        
                        if item_id:
                            matched_name = clean_name
                            # Assign directly without debug print
                            row[str(item_id)] = clean_val
                            if str(result_text).upper() in ['FAIL', 'F', '불량', 'NG', '부적합']:
                                row['합격여부'] = '불합격'
                                failed_items_list.append(matched_name or str(item_id))
                            break # Go to next row

            except Exception as e:
                print(f"Error processing {file}: {e}")
                
        if failed_items_list:
            row['불합격 항목'] = ", ".join(failed_items_list)
        else:
             row['합격여부'] = '합격' 
             has_data = any(pd.notna(row[str(i)]) for i in range(1, 28))
             if not has_data: row['합격여부'] = 'Unknown'
            
        df_row = pd.DataFrame([row])
        base_cols = ['일련번호', '품번', '시험일자']
        item_cols = [str(i) for i in range(1, 28)]
        extra_cols = ['합격여부', '불합격 항목', '운용월', 'Code', 'LOT 번호']
        all_cols = base_cols + item_cols + extra_cols
        
        for col in all_cols:
            if col not in df_row.columns: df_row[col] = np.nan
            
        return df_row[all_cols]
