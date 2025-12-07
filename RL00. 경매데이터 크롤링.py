

### 00. 경매데이터 크롤링


# pip install -U selenium webdriver-manager pandas beautifulsoup4 lxml openpyxl
import time, re, tempfile, shutil
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# ================== 설정 ==================
USER_ID = "아이디"
USER_PW = "비번"

LOGIN_URL = "http://sjau.co.kr/members/login.html"
LIST_URL_TPL = (
    "http://sjau.co.kr/auction/list.html"
    "?page={page}&listnum=100&orderby=&special=&court1=&damdang1=&syear=&sno="
    "&gamMin=0&gamMax=0&eng=99&uchal_min=&uchal_max="
    "&sday_s=2024-01-01&sday_e=2025-09-30"
    "&lowMin=0&lowMax=0&sido1=11&gugun1=&dong1=&bunji=&sagunname="
    "&barea_min=&barea_max=&larea_min=&larea_max="
    "&yongdo=01&gamratio=0&gongsiga=&people_min=&people_max=&addr="
)
START_PAGE = 1
END_PAGE   = 33   # ★ 여기 고정: 1~33페이지 수집
OUT_CSV  = "sjau_auction_cards_2425.csv"
OUT_XLSX = "sjau_auction_cards_2425.xlsx"

DBG = Path("debug_ssl"); DBG.mkdir(exist_ok=True)
# =========================================

# 임시 프로필(세션 유지)
profile_dir = tempfile.mkdtemp(prefix="sjau_profile_")

def dump(driver, name):
    (DBG / f"{name}.html").write_text(driver.page_source, encoding="utf-8", errors="ignore")
    try: driver.save_screenshot(str(DBG / f"{name}.png"))
    except: pass
    print(f"📝 dump: {name}.html / {name}.png")

def bypass_ssl_interstitial(driver):
    time.sleep(0.5)
    try:
        btn = driver.find_element(By.ID, "details-button"); btn.click(); time.sleep(0.3)
        go  = driver.find_element(By.ID, "proceed-link");  go.click();  time.sleep(0.5)
        return True
    except Exception:
        pass
    try:
        driver.execute_script(
            "document.body.innerHTML.indexOf('연결이 비공개로')>-1 && "
            "(document.getElementById('details-button')?.click(),"
            " document.getElementById('proceed-link')?.click());"
        )
        time.sleep(0.7)
        return True
    except:
        return False

options = webdriver.ChromeOptions()
options.add_argument("--headless=new")          # 문제 시 주석처리해 창 보이게
options.add_argument("--window-size=1440,900")
options.add_argument("--disable-gpu")
options.add_argument("--disable-extensions")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--remote-allow-origins=*")
options.add_argument(f"--user-data-dir={profile_dir}")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--allow-insecure-localhost")
options.add_argument("--allow-running-insecure-content")
options.add_argument("--unsafely-treat-insecure-origin-as-secure=http://sjau.co.kr")
options.set_capability("acceptInsecureCerts", True)
options.add_argument(
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 15)

def find_first(cands):
    last = None
    for by, sel in cands:
        try: return wait.until(EC.presence_of_element_located((by, sel)))
        except Exception as e: last = e
    if last: raise last

def scroll_to_bottom(driver, rounds=14, pause=0.7):
    last_h = 0
    for _ in range(rounds):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        h = driver.execute_script("return document.body.scrollHeight")
        if h == last_h: break
        last_h = h

def extract_cards_via_js(driver):
    js = r"""
    function getCards(){
      let nodes = Array.from(document.querySelectorAll(
        "div.list_box, li.list_box, div.card, li.card, div.auction_item, li.auction_item"
      ));
      if (nodes.length < 5) {
        const labs = Array.from(document.querySelectorAll("body *")).filter(el=>{
          try{ return (el.innerText||"").includes("감정가"); }catch(e){return false;}
        });
        const set = new Set();
        labs.forEach(el=>{
          let cur = el;
          for (let i=0;i<8 && cur && cur.parentElement;i++){
            cur = cur.parentElement;
            if(!cur) break;
            const txt = (cur.innerText||"").trim();
            const okPrice = (txt.includes("감정가")?1:0) + (txt.includes("최저가")?1:0) + (txt.includes("낙찰가")?1:0);
            const okInfo  = /서울|경기|인천|부산|대구|대전|광주|울산|세종|강원|충북|충남|전북|전남|경북|경남|㎡|평/.test(txt);
            if (okPrice >= 1 && okInfo && txt.length > 60){ set.add(cur); break; }
          }
        });
        nodes = Array.from(set);
      }
      return nodes.map(n=>({html: n.outerHTML, text: (n.innerText||"").trim()}));
    }
    return getCards();
    """
    try: return driver.execute_script(js)
    except: return []

def num_or_none(s):
    if not s: return None
    m = re.sub(r"[^\d]", "", str(s))
    return int(m) if m else None

def parse_card_text(t):
    """실거래 금액/비율, 낙찰비율까지 분리"""
    t2 = re.sub(r"[ \t]+", " ", t).strip()

# 사건번호
    m_case = re.search(r"([가-힣]+[0-9]+계)\s+(\d{4}[-.]\d+)", t2)
    court, case_no = (m_case.group(1), m_case.group(2)) if m_case else (None, None)
# 매각기일 (2024~2025)
    m_date = re.search(r"(2024|2025)[-\.]\d{2}[-\.]\d{2}", t2)
    sale_dt = m_date.group(0) if m_date else None
# 주소
    m_addr = re.search(r"(서울특별시)[^|]+", t2)
    address = m_addr.group(0).strip() if m_addr else None
    
# 건물/토지 면적
    def ffloat(x):
        try: return float(str(x).replace(",",""))
        except: return None
    m_b = re.search(r"(건물|전용)\s*([\d.,]+)\s*㎡", t2)
    building_m2 = ffloat(m_b.group(2)) if m_b else None
    m_l = re.search(r"토지\s*([\d.,]+)\s*㎡", t2)
    land_m2 = ffloat(m_l.group(1)) if m_l else None
# 감정가/최저가/낙찰가
    m_app = re.search(r"감정가\s*([\d,]+)", t2)
    m_low = re.search(r"최저가\s*([\d,]+)", t2)
    m_bid = re.search(r"낙찰가\s*([\d,]+)", t2)
# 실거래가 및 비율 / 낙찰비율
    m_real_ratio = re.search(r"실거래\((\d+)%\)", t2)
    m_real       = re.search(r"실거래(?:\([^)]*\))?\s*([\d,]+)", t2)
    m_bid_ratio  = re.search(r"낙찰\((\d+)%\)", t2)
    
    # 상태 (신건 / 유찰 / 낙찰 등) - 괄호("(") 앞까지만 매칭
    m_state = re.search(
    r"(신건|유찰\s*\n?\s*\d*회|낙찰|재진행|변경|매각|취하|정지|기각)\s*\(",
    t2
    )
    if m_state:
        state = m_state.group(1).replace("유찰", "유찰 ").replace("  ", " ").strip()
    else:
        state = None
    # 상태 뒤 괄호 안 숫자 비율 (예: 신건 (100%) → 100, 또는 다음 줄 괄호)
    m_state_ratio = re.search(
        r"(?:신건|유찰\s*\n?\s*\d*회|낙찰|재진행|변경|매각불허)[^\n<)]*(?:\n\s*)?\((\d+)%\)",
        t2,
        re.MULTILINE,
    )
    state_ratio = int(m_state_ratio.group(1)) if m_state_ratio else None

    return {
        "법원계": court,
        "사건번호": case_no,
        "매각기일": sale_dt,
        "소재지": address,
        "건물면적_㎡": building_m2,
        "토지면적_㎡": land_m2,
        "감정가": num_or_none(m_app.group(1)) if m_app else None,
        "최저가": num_or_none(m_low.group(1)) if m_low else None,
        "낙찰가": num_or_none(m_bid.group(1)) if m_bid else None,
        "낙찰가율": int(m_bid_ratio.group(1)) if m_bid_ratio else None,
        "실거래가": num_or_none(m_real.group(1)) if m_real else None,
        "실거래비율": int(m_real_ratio.group(1)) if m_real_ratio else None,
        "state": state,
        "state_ratio": state_ratio,
        "raw_text": t2
    }

# ================== 실행 ==================
try:
    # 1) 로그인
    driver.get(LOGIN_URL)
    time.sleep(1.0)
    dump(driver, "01_login_page_raw")
    if "연결이 비공개로" in driver.page_source or "ERR_CERT" in driver.page_source:
        print("⚠️ SSL 경고 감지 → 자동 우회 시도")
        bypass_ssl_interstitial(driver)
        time.sleep(0.8)
        dump(driver, "01_login_page_after_bypass")

    id_candidates = [
        (By.CSS_SELECTOR, "input[type='text'][name*='id']"),
        (By.CSS_SELECTOR, "input[type='email']"),
        (By.XPATH, "//input[contains(@placeholder,'아이디') and (@type='text' or @type='email')]"),
        (By.XPATH, "(//input[@type='text' or @type='email'])[1]"),
    ]
    pw_candidates = [
        (By.CSS_SELECTOR, "input[type='password']"),
        (By.XPATH, "//input[@type='password' and contains(@placeholder,'비밀번호')]"),
    ]
    def find_first(cands):
        last=None
        for by,sel in cands:
            try: return wait.until(EC.presence_of_element_located((by, sel)))
            except Exception as e: last=e
        if last: raise last

    id_box = find_first(id_candidates)
    pw_box = find_first(pw_candidates)
    id_box.clear(); id_box.send_keys(USER_ID)
    pw_box.clear(); pw_box.send_keys(USER_PW)

    btn = None
    for by, sel in [
        (By.XPATH, "//button[@type='submit' and contains(.,'로그인')]"),
        (By.XPATH, "//button[contains(.,'회원 로그인') or contains(.,'로그인')]"),
        (By.CSS_SELECTOR, "button[type='submit']"),
        (By.XPATH, "//input[@type='submit' and contains(@value,'로그인')]"),
    ]:
        try: btn = driver.find_element(by, sel); break
        except NoSuchElementException: pass
    if not btn:
        dump(driver, "02_no_login_button")
        raise TimeoutException("로그인 버튼을 찾지 못했습니다.")
    btn.click()

    try:
        wait.until(EC.presence_of_element_located(
            (By.XPATH, "//*[contains(.,'로그아웃') or contains(@class,'gnb') or contains(.,'경매')]")))
        print("✅ 로그인 성공")
    except TimeoutException:
        dump(driver, "03_after_login_timeout")
        raise

    # 2) 1~11페이지 고정 수집
    all_rows = []
    for page in range(START_PAGE, END_PAGE + 1):
        url = LIST_URL_TPL.format(page=page)
        print(f"-> 수집: page={page}  {url}")
        driver.get(url)
        time.sleep(1.0)
        scroll_to_bottom(driver, rounds=14, pause=0.7)

        dump(driver, f"list_page_{page}")
        cards = extract_cards_via_js(driver)
        print(f"  - 카드 후보 {len(cards)}개")

        if not cards:
            # 비어있어도 다음 페이지 진행(요청이 1~11 고정이므로)
            continue

        for c in cards:
            text = c.get("text") or ""
            if not text or len(text) < 20:
                soup = BeautifulSoup(c.get("html",""), "html.parser")
                text = soup.get_text(" ", strip=True)
            row = parse_card_text(text)
            row["page"] = page
            all_rows.append(row)

    if not all_rows:
        raise SystemExit("수집 결과가 없습니다. debug_ssl/list_page_*.html 확인 필요.")

    # 3) 정리/저장
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["법원계","사건번호","소재지"], keep="first")

    def safe_div(a,b):
        try: return a/b if (a and b) else None
        except: return None
    df["최저가율"] = df.apply(lambda r: safe_div(r["최저가"], r["감정가"]), axis=1)
    df["낙찰률"] = df.apply(lambda r: safe_div(r["낙찰가"], r["감정가"]), axis=1)

    import math
    def uchal_est(ratio):
        if not ratio or ratio <= 0: return None
        try:  return max(0, int(round(math.log(ratio / 100 if ratio > 1 else ratio) / math.log(0.8))))
        except Exception:
            return None
    df["유찰추정"] = df["최저가율"].map(uchal_est)
    df["유찰횟수"] = df["state_ratio"].map(uchal_est)

    # --- 주소 파생변수 ---
    # 1) add1: 소재지의 맨 윗줄만
    df["add1"] = df["소재지"].str.extract(r'^([^\r\n]*)', expand=False).str.strip()

    # 2) add2 및 3) 시도/시군구/읍면동/번지 분리
    #    예: "서울특별시 서대문구 연희동 739 ..." → 시도/시군구/읍면동/번지 추출
    addr_pat = r"^(?P<시도>[가-힣]+(?:특별시|광역시|특별자치시|특별자치도|도))\s+" \
               r"(?P<시군구>[가-힣]+(?:시|군|구))\s+" \
               r"(?P<읍면동>[가-힣0-9]+(?:읍|면|동))\s+" \
               r"(?P<번지>\d+(?:-\d+)?)"
    parts = df["add1"].str.extract(addr_pat)

    # 개별 컬럼 병합
    df[["시도","시군구","읍면동","번지"]] = parts[["시도","시군구","읍면동","번지"]]

    # add2: 시도~번지까지 조합
    df["add2"] = (
        df[["시도","시군구","읍면동","번지"]]
        .apply(lambda s: " ".join([x for x in s if pd.notnull(x) and str(x).strip() != ""]), axis=1)
        .replace("", pd.NA)
    )

    # 4) 층 정보 -------------------------------
    df["floor"] = (
        df["raw_text"]
        .astype(str)
        .str.extract(r"(\d+)\s*층", expand=False)
        .astype("Int64")
    )

    # 층구분: 5층 이하 = low / 6~14층 = mid / 15층 이상 = high
    s = pd.Series(pd.NA, index=df.index, dtype="string")  # 먼저 전부 NA (string dtype)
    s = s.mask(df["floor"].notna() & (df["floor"] <= 5),  "low")
    s = s.mask(df["floor"].notna() & (df["floor"] >= 15), "high")
    s = s.mask(df["floor"].notna() & (df["floor"].between(6, 14)), "mid")

    df["층구분"] = s  # floor가 NA인 곳은 그대로 <NA> 유지
    # --- 건물면적 구분 ---
    df["면적구분"] = np.select(
        [
            df["건물면적_㎡"].notna() & (df["건물면적_㎡"] <= 59),
            df["건물면적_㎡"].notna() & (df["건물면적_㎡"] <= 84),
            df["건물면적_㎡"].notna() & (df["건물면적_㎡"] <= 135),
            df["건물면적_㎡"].notna() & (df["건물면적_㎡"] > 135),
        ],
        ["소형", "중형", "대형", "초대형"],
        default=None
)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    df.to_excel(OUT_XLSX, index=False)
    print(f"✅ 저장 완료: {OUT_CSV} / {OUT_XLSX} (rows={len(df)})")

finally:
    driver.quit()
    shutil.rmtree(profile_dir, ignore_errors=True)


    

