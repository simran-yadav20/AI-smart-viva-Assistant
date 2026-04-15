import streamlit as st
import requests
import tempfile
from audio_recorder_streamlit import audio_recorder
import time
import base64

# ── Configuration ──────────────────────────────────────────────────────────────
BACKEND_URL = "http://127.0.0.1:8000"

# Lazy import for Whisper (only load when needed) with GPU acceleration
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        try:
            # Use GPU if available (RTX 4060)
            _whisper_model = whisper.load_model("tiny", device="cuda")
            print("✓ Whisper loaded on GPU (CUDA)")
        except:
            # Fallback to CPU if GPU not available
            _whisper_model = whisper.load_model("tiny", device="cpu")
            print("✓ Whisper loaded on CPU")
    return _whisper_model

def play_audio_silent(audio_bytes):
    """Play audio automatically without showing media controls"""
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay style="display: none;">
        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

st.set_page_config(
    page_title="AI Viva Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --error-color: #d62728;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-container h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .header-container p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    /* Question display */
    .question-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1.5rem 0;
    }
    
    .question-card h3 {
        margin-top: 0;
        color: #1f77b4;
    }
    
    /* Score display */
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0.5rem 0.5rem 0;
    }
    
    .score-good {
        background-color: #2ca02c;
        color: white;
    }
    
    .score-medium {
        background-color: #ff7f0e;
        color: white;
    }
    
    .score-low {
        background-color: #d62728;
        color: white;
    }
    
    /* Progress bar */
    .progress-text {
        font-weight: 600;
        color: #1f77b4;
        margin: 1rem 0 0.5rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background-color: #1a5fa0;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 5px;
        border: 2px solid #ddd;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1f77b4;
    }
    
    /* Metric styling */
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-top: 4px solid #1f77b4;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-box h3 {
        color: #666;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .metric-box .value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1>🎓 AI Smart Viva Assistant</h1>
    <p>Master your topics with AI-powered questions, evaluations, and real-time feedback</p>
</div>
""", unsafe_allow_html=True)

# Transcription Function
def transcribe(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    model = get_whisper_model()
    result = model.transcribe(path)
    return result["text"]

# Session Initialization
if "questions" not in st.session_state:
    st.session_state.questions = []
    st.session_state.current_q = 0
    st.session_state.scores = []
    st.session_state.show_correct = False
    st.session_state.show_evaluation = False
    st.session_state.question_audio = None
    st.session_state.improved_audio_url = None
    st.session_state.skip_audio_url = None
    st.session_state.question_spoken = False
    st.session_state.viva_started = False

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Session Control")
    
    if st.session_state.viva_started and st.session_state.questions:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Question", f"{st.session_state.current_q + 1}/{len(st.session_state.questions)}")
        with col2:
            if st.session_state.scores:
                avg = sum(st.session_state.scores) / len(st.session_state.scores)
                st.metric("Avg Score", f"{avg:.1f}/10")
        
        st.divider()
    
    st.markdown("### 🎯 Start New Viva")
    topic = st.text_input("Enter Topic", placeholder="e.g., Machine Learning, Python, etc.")
    
    col_start, col_reset = st.columns(2)
    with col_start:
        start_btn = st.button("🚀 Start Viva", use_container_width=True)
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            for key in st.session_state:
                if key.startswith('question') or key.startswith('current') or key.startswith('scores') or key.startswith('show') or key == 'viva_started':
                    if key == 'viva_started':
                        st.session_state[key] = False
                    elif isinstance(st.session_state[key], list):
                        st.session_state[key] = []
                    elif isinstance(st.session_state[key], bool):
                        st.session_state[key] = False
                    elif isinstance(st.session_state[key], int):
                        st.session_state[key] = 0
            st.rerun()
    
    st.divider()
    st.markdown("### 📊 Statistics")
    if st.session_state.scores:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(st.session_state.scores))
        with col2:
            st.metric("Average", f"{sum(st.session_state.scores)/len(st.session_state.scores):.1f}")
        with col3:
            st.metric("Best", max(st.session_state.scores))

# Main content
if start_btn and topic:
    if not topic:
        st.error("❌ Please enter a topic first!")
    else:
        try:
            with st.spinner("📚 Generating questions..."):
                response = requests.get(f"{BACKEND_URL}/generate/{topic}", timeout=30)
                if response.status_code == 200:
                    st.session_state.questions = response.json()["questions"]
                    st.session_state.current_q = 0
                    st.session_state.scores = []
                    st.session_state.show_correct = False
                    st.session_state.show_evaluation = False
                    st.session_state.question_audio = None
                    st.session_state.skip_audio_url = None
                    st.session_state.question_spoken = False
                    st.session_state.viva_started = True
                    st.success(f"✅ Generated {len(st.session_state.questions)} questions on {topic}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Backend error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend! Ensure backend and Ollama are running.")
        except requests.exceptions.Timeout:
            st.error("❌ Request timeout! Backend taking too long.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Viva Flow
if st.session_state.questions and st.session_state.viva_started:
    if st.session_state.current_q < len(st.session_state.questions):
        current_question = st.session_state.questions[st.session_state.current_q]

        # Question Header
        progress = st.session_state.current_q + 1
        total = len(st.session_state.questions)
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"<p class='progress-text'>Question {progress} of {total}</p>", unsafe_allow_html=True)
        with col2:
            st.progress(progress / total)
        with col3:
            if st.session_state.scores:
                avg_score = sum(st.session_state.scores) / len(st.session_state.scores)
                st.metric("Running Avg", f"{avg_score:.1f}")

        st.divider()

        # Question Display
        st.markdown(f"""
        <div class="question-card">
            <h3>Question {progress}</h3>
            <p style="font-size: 1.1rem; color: #333;">{current_question}</p>
        </div>
        """, unsafe_allow_html=True)

        # Auto-speak question if not yet spoken
        if not st.session_state.get("question_spoken", False):
            try:
                with st.spinner("🎧 Generating question audio..."):
                    speak_response = requests.post(
                        f"{BACKEND_URL}/speak/",
                        json={"question": current_question},
                        timeout=30
                    )

                    if speak_response.status_code == 200:
                        st.session_state.question_audio = speak_response.content
                        st.session_state.question_spoken = True
            except Exception as e:
                st.warning(f"Could not generate audio: {str(e)}")

        # Play Question Audio (silently auto-playing)
        if st.session_state.question_audio:
            play_audio_silent(st.session_state.question_audio)

        st.divider()

        # Answer Recording Section
        st.markdown("### 🎤 Record Your Answer")
        st.info("Click the microphone button below to record your answer. Speak clearly and concisely.")
        
        audio_bytes = audio_recorder()

        if audio_bytes and not st.session_state.show_evaluation:
            with st.spinner("🔄 Transcribing your answer..."):
                text_answer = transcribe(audio_bytes)

            st.markdown("#### 📝 Your Answer")
            st.markdown(f"> {text_answer}")

            col1, col2 = st.columns(2)
            with col1:
                submit_btn = st.button("✅ Submit Answer", use_container_width=True, key="submit_answer")
            with col2:
                st.button("⏭️ Skip", use_container_width=True, key="skip_here", disabled=True)

            if submit_btn:
                try:
                    with st.spinner("📊 Evaluating your answer..."):
                        response = requests.post(
                            f"{BACKEND_URL}/evaluate/",
                            json={
                                "question": current_question,
                                "answer": text_answer
                            },
                            timeout=60
                        )

                        result = response.json()

                        st.session_state.evaluation = result["evaluation"]
                        st.session_state.improved = result["improved_answer"]
                        st.session_state.score = result["score"]
                        st.session_state.improved_audio_url = result.get("improved_answer_audio_url")
                        st.session_state.show_evaluation = True
                        st.session_state.scores.append(result["score"])
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error evaluating answer: {str(e)}")

        # Show Evaluation
        if st.session_state.show_evaluation:
            st.divider()
            st.markdown("### 📊 Evaluation Results")
            
            # Score Display
            score = st.session_state.score
            if score >= 7:
                score_class = "score-good"
                reaction = "Excellent! 🌟"
            elif score >= 5:
                score_class = "score-medium"
                reaction = "Good effort! 👍"
            else:
                score_class = "score-low"
                reaction = "Keep improving! 💪"

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: center;">
                    <span class="score-badge {score_class}">Score: {score}/10</span>
                    <p style="margin-top: 0.5rem; font-size: 1.1rem;">{reaction}</p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Evaluation")
                st.markdown(st.session_state.evaluation)
            
            with col2:
                st.markdown("#### ✨ Improved Answer")
                st.markdown(st.session_state.improved)
                
                if st.session_state.get("improved_audio_url"):
                    try:
                        audio_response = requests.get(st.session_state.improved_audio_url, timeout=10)
                        if audio_response.status_code == 200:
                            st.audio(audio_response.content, format="audio/wav", autoplay=True)
                    except Exception as e:
                        st.warning(f"Could not play audio: {str(e)}")

            st.divider()

            next_col1, next_col2, next_col3 = st.columns([1, 1, 2])
            with next_col1:
                if st.button("⏭️ Next Question", use_container_width=True, key="next_after_eval"):
                    st.session_state.question_audio = None
                    st.session_state.improved_audio_url = None
                    st.session_state.skip_audio_url = None
                    st.session_state.show_evaluation = False
                    st.session_state.question_spoken = False
                    st.session_state.current_q += 1
                    st.rerun()

        # Skip Logic
        if not st.session_state.show_evaluation and not audio_bytes:
            col1, col2 = st.columns(2)
            with col2:
                if st.button("⏭️ Skip Question", use_container_width=True, key="skip_btn"):
                    try:
                        with st.spinner("📚 Getting correct answer..."):
                            response = requests.post(
                                f"{BACKEND_URL}/skip/",
                                json={"question": current_question},
                                timeout=60
                            )

                            result = response.json()
                            correct = result["correct_answer"]
                            audio_url = result.get("audio_url")

                            st.session_state.correct_answer = correct
                            st.session_state.skip_audio_url = audio_url
                            st.session_state.scores.append(0)
                            st.session_state.show_correct = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error skipping question: {str(e)}")

        if st.session_state.get("show_correct", False):
            st.divider()
            st.markdown("### 📖 Correct Answer")
            st.markdown(f"> {st.session_state.correct_answer}")

            # Play audio with controls and autoplay
            if st.session_state.get("skip_audio_url"):
                try:
                    audio_response = requests.get(st.session_state.skip_audio_url, timeout=10)
                    if audio_response.status_code == 200:
                        st.audio(audio_response.content, format="audio/wav", autoplay=True)
                except Exception as e:
                    st.warning(f"Could not play audio: {str(e)}")

            if st.button("⏭️ Next Question", use_container_width=True, key="next_after_skip"):
                st.session_state.question_audio = None
                st.session_state.show_correct = False
                st.session_state.skip_audio_url = None
                st.session_state.question_spoken = False
                st.session_state.current_q += 1
                st.rerun()

    else:
        # Final Summary
        st.divider()
        st.markdown("""
        <div class="header-container" style="text-align: center; background: linear-gradient(135deg, #2ca02c 0%, #1f77b4 100%);">
            <h2>🎉 Viva Completed!</h2>
            <p>Congratulations on finishing the assessment!</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.scores:
            avg_score = sum(st.session_state.scores) / len(st.session_state.scores)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Total Questions</h3>
                    <div class="value">{len(st.session_state.scores)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Average Score</h3>
                    <div class="value">{avg_score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Best Score</h3>
                    <div class="value">{max(st.session_state.scores)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                worst = min(st.session_state.scores)
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Lowest Score</h3>
                    <div class="value">{worst}</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.markdown("### 📈 Detailed Results")
            
            # Score breakdown
            cols = st.columns(3)
            excellent = sum(1 for s in st.session_state.scores if s >= 7)
            good = sum(1 for s in st.session_state.scores if 5 <= s < 7)
            needs_work = sum(1 for s in st.session_state.scores if s < 5)
            
            with cols[0]:
                st.metric("Excellent (7+)", excellent)
            with cols[1]:
                st.metric("Good (5-6)", good)
            with cols[2]:
                st.metric("Needs Work (<5)", needs_work)
            
            # Recommendation
            st.divider()
            if avg_score >= 7:
                st.success("🌟 Excellent performance! You have a strong grasp of this topic.")
            elif avg_score >= 5:
                st.info("👍 Good job! With a bit more practice, you'll master this topic.")
            else:
                st.warning("💪 Keep practicing! More focused study will help improve your scores.")

        st.divider()
        if st.button("🔄 Start New Viva", use_container_width=True):
            for key in st.session_state:
                if key != 'viva_started':
                    if isinstance(st.session_state[key], list):
                        st.session_state[key] = []
                    elif isinstance(st.session_state[key], bool):
                        st.session_state[key] = False
                    elif isinstance(st.session_state[key], int):
                        st.session_state[key] = 0
            st.session_state.viva_started = False
            st.rerun()

elif not st.session_state.viva_started:
    st.markdown("<div style='text-align:center; padding: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("## 🚀 Welcome to AI Smart Viva Assistant")
    st.markdown("##### Ready to master any topic?")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown("""
<div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem; border-radius: 10px; text-align: left;">
<h3 style="text-align:center;">📋 How It Works</h3>
<ol>
<li><strong>Enter a topic</strong> you want to master</li>
<li><strong>AI generates questions</strong> tailored to that topic</li>
<li><strong>Record your answers</strong> using your microphone</li>
<li><strong>Get instant feedback</strong> with scores and improvements</li>
<li><strong>Review results</strong> and track your progress</li>
</ol>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.info("👈 Use the sidebar to get started!")

