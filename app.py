                        # Feedback suggestions
                        st.markdown("#### 💡 Feedback Suggestions")
                        for standard, suggestion in ct_suggest.items():
                            score = ct_scores.get(standard, 0)
                            if score < 0.6:
                                st.info(f"**{standard}:** {suggestion}")
        
        with tab3:
            # Enhanced CT analysis with highlighting guide
            st.subheader("Critical Thinking Analysis")
            
            # Color legend
            st.markdown("#### 🎨 CT Standards Color Guide")
            cols = st.columns(3)
            for idx, (standard, data) in enumerate(PAUL_CT_RUBRIC.items()):
                with cols[idx % 3]:
                    st.markdown(
                        f'<div style="padding: 0.5rem; margin: 0.25rem 0; border-radius: 5px; border-left: 4px solid {data["color"]};">'
                        f'<strong>{standard}</strong><br>'
                        f'<small>{data["description"][:100]}...</small>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
            
            # CT scores comparison across submissions
            st.markdown("#### 📊 CT Scores Comparison")
            ct_df_data = []
            for i, (meta, ct_scores) in enumerate(zip(submissions, ct_scores_all)):
                row = {"Filename": meta["filename"]}
                row.update(ct_scores)
                ct_df_data.append(row)
            
            if ct_df_data:
                ct_df = pd.DataFrame(ct_df_data)
                melted_df = ct_df.melt(id_vars=["Filename"], var_name="CT Standard", value_name="Score")
                fig = px.box(melted_df, x="CT Standard", y="Score", title="Distribution of CT Scores Across Standards")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed table
                st.dataframe(ct_df.set_index("Filename").round(3), use_container_width=True)
        
        with tab4:
            # Export section
            st.subheader("📤 Export Results")
            
            # Create comprehensive DataFrame
            df_summary = pd.DataFrame(rows)
            
            # Display preview
            st.markdown("#### Preview of Export Data")
            st.dataframe(df_summary[["filename", "word_count", "avg_ct_score", "fused_conf", "text_preview"]])
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv_bytes = df_summary.to_csv(index=False).encode("utf-8")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "📥 Download CSV", 
                    data=csv_bytes, 
                    file_name=f"ctlearner_results_{timestamp}.csv", 
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel Export
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                    # Main results
                    df_summary.to_excel(writer, index=False, sheet_name="Results")
                    
                    # CT scores detailed
                    ct_details = []
                    for i, (meta, ct_scores, ct_suggest) in enumerate(zip(submissions, ct_scores_all, ct_suggestions_all)):
                        for standard, score in ct_scores.items():
                            ct_details.append({
                                "Filename": meta["filename"],
                                "CT_Standard": standard,
                                "Score": score,
                                "Suggestion": ct_suggest[standard]
                            })
                    pd.DataFrame(ct_details).to_excel(writer, index=False, sheet_name="CT_Details")
                    
                    # Emotion scores
                    emotion_details = []
                    for i, (meta, emotion_scores) in enumerate(zip(submissions, fused_results)):
                        for emotion, score in emotion_scores[3].items():
                            emotion_details.append({
                                "Filename": meta["filename"],
                                "Emotion": emotion,
                                "Score": score
                            })
                    pd.DataFrame(emotion_details).to_excel(writer, index=False, sheet_name="Emotion_Details")
                
                st.download_button(
                    "📊 Download Excel", 
                    data=towrite.getvalue(), 
                    file_name=f"ctlearner_results_{timestamp}.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Additional export options
            st.markdown("#### Additional Reports")
            
            # CT Improvement Report
            if st.button("📋 Generate CT Improvement Report", use_container_width=True):
                improvement_data = []
                for meta, ct_scores in zip(submissions, ct_scores_all):
                    weak_areas = [std for std, score in ct_scores.items() if score < 0.6]
                    strong_areas = [std for std, score in ct_scores.items() if score >= 0.7]
                    
                    improvement_data.append({
                        "Filename": meta["filename"],
                        "Overall_CT_Score": np.mean(list(ct_scores.values())),
                        "Weak_Areas": ", ".join(weak_areas) if weak_areas else "None",
                        "Strong_Areas": ", ".join(strong_areas) if strong_areas else "None",
                        "Priority_Level": "High" if len(weak_areas) > 3 else "Medium" if len(weak_areas) > 1 else "Low"
                    })
                
                improvement_df = pd.DataFrame(improvement_data)
                st.dataframe(improvement_df, use_container_width=True)
                
                # Download improvement report
                csv_improvement = improvement_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Improvement Report", 
                    data=csv_improvement, 
                    file_name=f"ct_improvement_report_{timestamp}.csv", 
                    mime="text/csv"
                )

        st.success("🎉 Analysis complete! Explore the results in the tabs above.")
        
    else:
        # Welcome state - show when no analysis has been run
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🚀 Getting Started")
            st.markdown("""
            1. **Upload** student submissions (TXT, PDF, or DOCX)
            2. **Configure** analysis settings in the sidebar
            3. **Click** 'Start Analysis' to begin processing
            4. **Explore** results in the interactive dashboard
            
            ### 📊 What You'll Get:
            - **Emotion Analysis**: AI-powered emotion detection with explainable triggers
            - **Critical Thinking Assessment**: Automated scoring using Paul's Rubric
            - **Sentence Highlighting**: Visual indicators of CT standards in text
            - **Interactive Visualizations**: Charts and graphs for data insights
            - **Exportable Reports**: CSV and Excel downloads for further analysis
            """)
        
        with col2:
            st.subheader("🎯 CT Standards Covered")
            for standard in list(PAUL_CT_RUBRIC.keys())[:5]:
                st.markdown(f"✅ **{standard}**")
            if len(PAUL_CT_RUBRIC) > 5:
                with st.expander("See all standards"):
                    for standard in list(PAUL_CT_RUBRIC.keys())[5:]:
                        st.markdown(f"✅ **{standard}**")
            
            st.subheader("😊 Emotions Detected")
            emotions_display = ", ".join(EKMAN_PLUS[:4]) + ", ..."
            st.markdown(f"`{emotions_display}`")

# Run the app
if __name__ == "__main__":
    main()
