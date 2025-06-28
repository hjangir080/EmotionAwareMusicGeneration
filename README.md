# Emotion-Aware Music Generation from Literature

 - All files are in Python notebook format, each capturing a checkpoint of our progress.
 - Notebooks dl_musicgen.ipynb and v1.ipynb include audio output generation


This project focuses on generating music that reflects the emotional journey within a given text, leveraging deep learning and natural language processing. It integrates text emotion analysis inspired by TransProse with cutting-edge music generation models (like MusicGen and MusicLM-style architectures) to create music that dynamically aligns with a narrative's emotional shifts.

## Key Features

  * **Emotion Extraction:** Accurately extracts nuanced, segment-wise emotion vectors from literary texts using a RoBERTa-based classifier.
  * **Dynamic Emotion-to-Music Mapping:** Converts emotional cues into musical parameters (e.g., tempo, key, rhythm) using a neural mapping combined with expert rules, ensuring musical alignment with evolving narratives.
  * **Temporal Coherence:** Utilizes sequence modeling techniques (like LSTMs) to ensure smooth, emotionally consistent transitions between musical segments.
  * **Prompt-Based Music Generation:** Generates descriptive text prompts from emotional maps, which are then used by open-source neural models (e.g., MusicGen) to synthesize audio.
  * **Interactive System:** Includes an interactive notebook interface for text input, emotional arc visualization, segment playback, and MIDI file saving/combination.

## Problem Addressed

Current emotion-aware music generation often suffers from static emotion mapping, misalignment with dynamic narratives, and high computational demands. This project aims to overcome these limitations by creating a system that produces truly narrative-aligned music efficiently.

## How It Works (Methodology Overview)

The system follows an end-to-end pipeline:

1.  **Text Processing:** Cleans and segments literary texts into narrative units.
2.  **Emotion Extraction:** Determines emotion scores and intensity for each text segment, creating an emotional map.
3.  **Emotion-to-Music Mapping:** Translates emotional data into musical parameters and generates structured prompts for music models.
4.  **Music Generation:** Utilizes pre-trained models like MusicGen to synthesize audio based on the emotion-conditioned prompts.
5.  **Temporal Coherence:** Ensures smooth musical transitions reflecting the narrative's emotional flow.
