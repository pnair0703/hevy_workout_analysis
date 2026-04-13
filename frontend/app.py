"""IronAgent — Streamlit Frontend.

Usage:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import run_agent
from src.tools.hevy import HevyClient
from src.tools.volume_calc import compute_training_analysis
from src.tools.anomaly import detect_anomalies
from src.tools.nutrition import check_nutrition_constraints, adjust_volume_for_phase
from src.models.program import UserProfile, TrainingPhase


# ── Page config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="IronAgent",
    page_icon="🏋️",
    layout="wide",
)

st.title("🏋️ IronAgent")
st.caption("Agentic AI training coach powered by your Hevy data")


# ── Sidebar: User profile ──────────────────────────────────────────

with st.sidebar:
    st.header("Your Profile")

    bodyweight = st.number_input("Bodyweight (lbs)", value=171, step=1)
    calories = st.number_input("Daily calories", value=2500, step=50)
    protein = st.number_input("Protein (g)", value=200, step=5)
    carbs = st.number_input("Carbs (g)", value=280, step=5)
    fat = st.number_input("Fat (g)", value=60, step=5)
    phase = st.selectbox("Training phase", ["cut", "bulk", "maintain", "peak"])
    training_days = st.slider("Training days/week", 3, 7, 5)
    injuries = st.text_input("Current injuries (comma-separated)", "")

    profile = UserProfile(
        name="User",
        bodyweight_lbs=bodyweight,
        calories=calories,
        protein_g=protein,
        carbs_g=carbs,
        fat_g=fat,
        phase=TrainingPhase(phase),
        training_days_per_week=training_days,
        injuries=[i.strip() for i in injuries.split(",") if i.strip()],
    )

    st.divider()
    st.caption("Data from Hevy API")


# ── Dashboard tab + Chat tab ────────────────────────────────────────

tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard"])


# ── Dashboard ───────────────────────────────────────────────────────

with tab_dashboard:
    @st.cache_data(ttl=300)
    def load_dashboard_data():
        with HevyClient() as client:
            workouts = client.get_recent_workouts(days=28)
            all_workouts = client.get_all_workouts(max_pages=10)
            templates = client.build_template_lookup()
        analysis = compute_training_analysis(workouts, templates, period_days=28)
        anomalies = detect_anomalies(all_workouts, templates, n_weeks=12)
        return workouts, templates, analysis, anomalies

    try:
        workouts, templates, analysis, anomalies = load_dashboard_data()

        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Workouts (28d)", analysis.total_workouts)
        col2.metric("Avg Duration", f"{analysis.avg_duration_minutes or 0:.0f} min")
        col3.metric("Muscle Gaps", len(analysis.gaps))

        iso = anomalies.get("pattern_anomaly", {})
        col4.metric(
            "Pattern Status",
            "⚠️ Abnormal" if iso.get("is_anomalous") else "✅ Normal"
        )

        # Volume chart
        st.subheader("Weekly Volume by Muscle Group")
        volume_data = {
            v.muscle_group: v.weekly_sets
            for v in analysis.volume_by_muscle
        }
        st.bar_chart(volume_data)

        # Gaps
        if analysis.gaps:
            st.subheader("🚨 Training Gaps (below MEV)")
            for gap in analysis.gaps:
                st.warning(f"**{gap}** — below minimum effective volume")

        # Anomaly detection
        st.subheader("🔬 Anomaly Detection")

        z_anomalies = anomalies.get("individual_anomalies", [])
        if z_anomalies:
            st.write("**Z-score flags (vs your personal baseline):**")
            for a in z_anomalies:
                severity_color = {"high": "🔴", "moderate": "🟡", "low": "🟠"}
                st.write(f"{severity_color.get(a['severity'], '?')} {a['description']}")
        else:
            st.success("No individual muscle anomalies detected.")

        iso_result = anomalies.get("pattern_anomaly", {})
        if iso_result.get("is_anomalous"):
            st.warning(f"**Isolation Forest:** {iso_result['description']}")
        else:
            st.success(f"**Isolation Forest:** {iso_result.get('description', 'Normal')}")

        # Recent workouts
        st.subheader("📋 Recent Workouts")
        for w in workouts[:5]:
            with st.expander(f"{w.start_time.strftime('%Y-%m-%d')} — {w.title} ({w.duration_minutes or '?'}min)"):
                for ex in w.exercises:
                    top = ex.top_set
                    if top and top.weight_lbs:
                        st.write(
                            f"• **{ex.title}**: {ex.num_working_sets} sets, "
                            f"top set {top.weight_lbs} lbs × {top.reps}"
                        )
                    else:
                        st.write(f"• **{ex.title}**: {ex.num_working_sets} sets")

    except Exception as e:
        st.error(f"Failed to load Hevy data: {e}")
        st.info("Make sure HEVY_API_KEY is set in your .env file.")


# ── Chat ────────────────────────────────────────────────────────────

with tab_chat:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("meta"):
                st.caption(message["meta"])

    # Chat input
    if prompt := st.chat_input("Ask about your training..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your training data..."):
                result = run_agent(prompt)

            answer = result.get("answer", "Sorry, I couldn't generate a response.")
            model_used = result.get("model_used", "unknown")
            query_type = result.get("query_type", "unknown")

            st.markdown(answer)
            meta = f"Route: {query_type} → {model_used}"
            st.caption(meta)

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": meta,
        })