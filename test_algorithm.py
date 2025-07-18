#!/usr/bin/env python3
"""
Test script to demonstrate the CSV-based song recommendation algorithm
"""

import pandas as pd
import random
import os

def load_emotion_songs(emotion):
    """Load songs from CSV file based on detected emotion"""
    emotion_lower = emotion.lower()
    csv_file = f"songs/{emotion_lower}.csv"
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return []
    
    try:
        df = pd.read_csv(csv_file)
        # Convert DataFrame to list of dictionaries
        songs = []
        for _, row in df.iterrows():
            song = {
                'name': row['Name'],
                'album': row['Album'],
                'artist': row['Artist']
            }
            songs.append(song)
        print(f"Loaded {len(songs)} songs for emotion: {emotion}")
        return songs
    except Exception as e:
        print(f"Error loading CSV file {csv_file}: {e}")
        return []

def select_songs_by_preferences(songs, language="", artist=""):
    """Filter songs based on user preferences"""
    filtered_songs = songs
    
    if artist:
        # Filter by artist (case-insensitive)
        filtered_songs = [song for song in filtered_songs 
                         if artist.lower() in song['artist'].lower()]
        print(f"Filtered to {len(filtered_songs)} songs by artist: {artist}")
    
    return filtered_songs

def simulate_youtube_search(song_name, artist_name):
    """Simulate YouTube search (for testing purposes)"""
    return {
        "title": f"{song_name} - {artist_name}",
        "url": f"https://www.youtube.com/results?search_query={song_name}+{artist_name}+official",
        "platform": "YouTube",
        "original_song": song_name,
        "original_artist": artist_name
    }

def recommend_songs(emotion, artist="", max_recommendations=5):
    """Main recommendation algorithm"""
    print(f"\n=== Song Recommendation Algorithm ===")
    print(f"Detected Emotion: {emotion}")
    print(f"Artist Filter: {artist if artist else 'None'}")
    print(f"Max Recommendations: {max_recommendations}")
    
    # Step 1: Load songs from CSV based on emotion
    print(f"\nStep 1: Loading songs from {emotion.lower()}.csv...")
    songs_from_csv = load_emotion_songs(emotion)
    
    if not songs_from_csv:
        print(f"No songs found for emotion: {emotion}")
        return []
    
    # Step 2: Filter songs based on user preferences
    print(f"\nStep 2: Filtering songs by preferences...")
    filtered_songs = select_songs_by_preferences(songs_from_csv, artist=artist)
    
    if not filtered_songs:
        print(f"No songs match the artist filter: {artist}")
        return []
    
    # Step 3: Select random songs from the filtered list
    print(f"\nStep 3: Selecting random songs...")
    if len(filtered_songs) > max_recommendations:
        selected_songs = random.sample(filtered_songs, max_recommendations)
    else:
        selected_songs = filtered_songs
    
    print(f"Selected {len(selected_songs)} songs from CSV")
    
    # Step 4: Simulate YouTube search for each selected song
    print(f"\nStep 4: Searching songs on YouTube...")
    recommendations = []
    for i, song in enumerate(selected_songs, 1):
        print(f"  {i}. Searching: {song['name']} by {song['artist']}")
        youtube_result = simulate_youtube_search(song['name'], song['artist'])
        recommendations.append(youtube_result)
    
    return recommendations

def main():
    """Test the algorithm with different scenarios"""
    
    # Test 1: Happy emotion, no artist filter
    print("=" * 60)
    print("TEST 1: Happy emotion, no artist filter")
    print("=" * 60)
    recommendations = recommend_songs("Happy")
    print(f"\nFinal Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['title']}")
        print(f"     URL: {rec['url']}")
    
    # Test 2: Sad emotion, filtered by artist
    print("\n" + "=" * 60)
    print("TEST 2: Sad emotion, filtered by 'Billie Eilish'")
    print("=" * 60)
    recommendations = recommend_songs("Sad", artist="Billie Eilish")
    print(f"\nFinal Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['title']}")
        print(f"     URL: {rec['url']}")
    
    # Test 3: Angry emotion, filtered by artist
    print("\n" + "=" * 60)
    print("TEST 3: Angry emotion, filtered by 'Eminem'")
    print("=" * 60)
    recommendations = recommend_songs("Angry", artist="Eminem")
    print(f"\nFinal Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['title']}")
        print(f"     URL: {rec['url']}")

if __name__ == "__main__":
    main() 