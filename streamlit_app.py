# ---------- MODEL SELECTION & PERFORMANCE ----------
st.subheader("📈 Model Selection & Performance")

# Show performance as a DataFrame
results_df = pd.DataFrame(list(results.items()), columns=["Model", "R² Score"]).sort_values(by="R² Score", ascending=False)
st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

# Visual comparison
st.bar_chart(results_df.set_index("Model"))

# Best performing model
best_model_name = results_df.iloc[0]['Model']
best_r2_score = results_df.iloc[0]['R² Score']
st.markdown(f"✅ **Best Performing Model:** `{best_model_name}` with an R² score of **{best_r2_score:.3f}**")

# User-selectable model
selected_model_name = st.selectbox("Select a model for prediction", results_df["Model"].tolist())
selected_model = trained_models[selected_model_name]
st.markdown(f"*R² Score* for `{selected_model_name}`: **{results[selected_model_name]:.3f}**")
