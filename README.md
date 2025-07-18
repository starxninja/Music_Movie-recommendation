﻿# Music_Movie-recommendation
MoodMelody: Emotion-Based Music & Movie Recommendation System
Overview
MoodMelody is a web application that recommends music and movies based on your detected facial emotion. Using your webcam, a deep learning model classifies your mood, then suggests curated songs (from CSV files) or movies (via YouTube search) tailored to your current emotion. The system supports artist/language filtering and provides direct YouTube links for each recommendation.
Features
Real-Time Emotion Detection: Uses your webcam and a CNN model to classify emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised.
Curated Song Recommendations: Selects songs from emotion-specific CSV files, filtered by your preferences (artist/language).
YouTube Integration: Searches for each recommended song or movie on YouTube and returns direct video links.
Web Interface: Modern, responsive UI for easy interaction.
Movie Recommendations: Suggests movies based on your mood using YouTube search.
Test Script: Standalone script to test the recommendation algorithm without the web interface.
Model Training: Includes code to retrain the emotion detection model on your own dataset.
