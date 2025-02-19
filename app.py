import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import analysis
from visualization.plot_joint_angles import plot_joint_angles

SAVE_DIR = "saved_results"
os.makedirs(SAVE_DIR, exist_ok=True)

st.title("肩関節外転 動作分析アプリ")

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "avi", "mov"])

if uploaded_file:
    video_path = os.path.join(SAVE_DIR, "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("動画を解析中...")
    output_video_path = os.path.join(SAVE_DIR, "processed_video.mp4")
    angles_data = analysis.process_video_with_markers(video_path, output_video_path)

    max_rom = analysis.get_maximum_range_of_motion(angles_data)
    st.write(f"最大可動域（ROM）: {max_rom:.2f}°")

    fig = plot_joint_angles(angles_data)
    graph_path = os.path.join(SAVE_DIR, "angle_plot.png")
    fig.savefig(graph_path)

    st.pyplot(fig)

    st.write("📹 関節マーカー付きの解析動画")
    st.video(output_video_path)

    csv_path = os.path.join(SAVE_DIR, "angle_data.csv")
    df = pd.DataFrame({"Frame": range(len(angles_data)), "Angle": angles_data})
    df.to_csv(csv_path, index=False)

    st.write("📁 データのダウンロード:")
    st.download_button("CSVをダウンロード", data=open(csv_path, "rb"), file_name="angle_data.csv", mime="text/csv")
    st.download_button("グラフをダウンロード", data=open(graph_path, "rb"), file_name="angle_plot.png", mime="image/png")
    st.download_button("動画をダウンロード", data=open(output_video_path, "rb"), file_name="processed_video.mp4", mime="video/mp4")

