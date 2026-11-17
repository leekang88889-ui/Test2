import os

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

st.set_page_config(page_title="AI 번역기", page_icon="🌐", layout="centered")

load_dotenv()

MAX_CHARS = 5000

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into {target_language}. "
    "Output only the translated text, with no explanations, quotes, or notes. "
    "Preserve line breaks, tone, and formatting."
)

LANGUAGES = {
    "영어": "English",
    "일본어": "Japanese",
    "베트남어": "Vietnamese",
}


def get_config(name: str, default: str | None = None) -> str | None:
    """환경변수 우선, 없으면 st.secrets, 둘 다 없으면 default."""
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name)
    except Exception:  # secrets.toml 이 없는 경우
        value = None
    return value or default


API_KEY = get_config("OPENAI_API_KEY")
MODEL = get_config("OPENAI_MODEL", "gpt-6.1-sol")
BASE_URL = get_config("OPENAI_BASE_URL")


@st.cache_resource
def get_client(api_key: str, base_url: str | None) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url or None, timeout=60.0)


class TranslationError(Exception):
    """사용자에게 그대로 보여줘도 되는 한국어 메시지를 담는 예외."""


def translate(text: str, target_lang: str) -> str:
    client = get_client(API_KEY, BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            # gpt-6.1-sol 은 temperature 를 기본값(1)만 지원하므로 지정하지 않는다.
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(target_language=target_lang)},
                {"role": "user", "content": text},
            ],
        )
    except AuthenticationError:
        raise TranslationError("API 키가 올바르지 않습니다. 설정(.env 또는 Secrets)을 확인해 주세요.")
    except PermissionDeniedError:
        raise TranslationError("이 API 키로는 요청할 권한이 없습니다. 키의 권한을 확인해 주세요.")
    except RateLimitError:
        raise TranslationError("요청 한도를 초과했거나 사용량이 부족합니다. 잠시 후 다시 시도해 주세요.")
    except NotFoundError:
        raise TranslationError(f"모델 '{MODEL}'을(를) 찾을 수 없거나 사용할 수 없습니다. OPENAI_MODEL 값을 확인해 주세요.")
    except BadRequestError:
        raise TranslationError("요청이 거부되었습니다. 모델이 지원하지 않는 옵션이거나 입력 내용에 문제가 있을 수 있습니다.")
    except APITimeoutError:
        raise TranslationError("응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.")
    except APIConnectionError:
        raise TranslationError("서버에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요.")
    except Exception:
        raise TranslationError("번역 중 알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")

    content = resp.choices[0].message.content if resp.choices else None
    if not content or not content.strip():
        raise TranslationError("번역 결과가 비어 있습니다. 다시 시도해 주세요.")
    return content.strip()


def handle_click(text: str, label: str, lang: str) -> None:
    """검증 후 번역하고 결과를 session_state 에 저장한다."""
    st.session_state.error = None
    st.session_state.warning = None

    if not text.strip():
        st.session_state.warning = "번역할 내용을 입력해 주세요."
        return
    if len(text) > MAX_CHARS:
        st.session_state.warning = f"입력은 최대 {MAX_CHARS:,}자까지 가능합니다. (현재 {len(text):,}자)"
        return

    try:
        with st.spinner(f"{label}로 번역 중..."):
            st.session_state.result = translate(text, lang)
        st.session_state.result_label = label
    except TranslationError as e:
        st.session_state.error = str(e)


# ---------------------------------------------------------------- UI
st.session_state.setdefault("result", None)
st.session_state.setdefault("result_label", None)
st.session_state.setdefault("error", None)
st.session_state.setdefault("warning", None)

# 푸터/배포 버튼만 숨기는 최소 CSS
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .block-container {padding-top: 2.5rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌐 AI 번역기")
st.caption("번역할 글을 입력하고 언어를 선택하세요. 입력 언어는 자동으로 감지됩니다.")

if not API_KEY:
    st.error(
        "OPENAI_API_KEY가 설정되지 않았습니다. 로컬에서는 `.env` 파일에, "
        "배포 환경에서는 Secrets에 키를 등록한 뒤 다시 실행해 주세요."
    )

text = st.text_area(
    "원문",
    height=200,
    placeholder="여기에 번역할 글을 입력하세요...",
    label_visibility="collapsed",
)
count = len(text)
if count > MAX_CHARS:
    st.markdown(f"<div style='text-align:right; color:#e5484d; font-size:.85rem'>{count:,} / {MAX_CHARS:,}자 (초과)</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div style='text-align:right; opacity:.6; font-size:.85rem'>{count:,} / {MAX_CHARS:,}자</div>", unsafe_allow_html=True)

cols = st.columns(len(LANGUAGES))
for col, (label, lang) in zip(cols, LANGUAGES.items()):
    if col.button(
        label,
        icon=":material/translate:",
        type="primary",
        use_container_width=True,
        disabled=not API_KEY,
    ):
        handle_click(text, label, lang)

if st.session_state.warning:
    st.warning(st.session_state.warning)
if st.session_state.error:
    st.error(st.session_state.error)

st.divider()
st.subheader("번역 결과")
with st.container(border=True):
    if st.session_state.result:
        st.caption(f"{st.session_state.result_label}로 번역됨")
        st.code(st.session_state.result, language=None, wrap_lines=True)
    else:
        st.caption("언어 버튼을 누르면 번역 결과가 여기에 표시됩니다.")
