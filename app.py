from flask import Flask, render_template, Response, jsonify, request
import gunicorn
from camera import VideoCamera, show_text, emotion_dict
import random
import pandas as pd
import os
from googleapiclient.discovery import build

app = Flask(__name__)

# YouTube API setup
YOUTUBE_API_KEY = "AIzaSyCuUW9HgGlgcFgYlMIgJVOW7zTMp16fVtk"
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# No initial song recommendations from CSV
headings = ("Name", "Album", "Artist")  # Kept for template compatibility

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
    
    # Note: Language filtering would require additional data in CSV
    # For now, we'll use all songs from the emotion category
    
    return filtered_songs

def search_song_on_youtube(song_name, artist_name):
    """Search a specific song on YouTube"""
    try:
        # Create search query with song name and artist
        query = f"{song_name} {artist_name} official"
        print(f"Searching YouTube for: {query}")
        
        # Search YouTube
        yt_request = youtube.search().list(
            part="id,snippet",
            q=query,
            type="video",
            maxResults=1,  # Get only the best match
            videoCategoryId="10"  # Music category
        )
        response = yt_request.execute()
        
        if response["items"]:
            item = response["items"][0]
            return {
                "title": f"{song_name} - {artist_name}",
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "platform": "YouTube",
                "original_song": song_name,
                "original_artist": artist_name
            }
        else:
            print(f"No YouTube results found for: {query}")
            return None
            
    except Exception as e:
        print(f"Error searching YouTube for {song_name}: {e}")
        return None

@app.route('/')
def index():
    print("Serving index.html with no initial data")
    return render_template('index.html', headings=headings, data=[])

def gen(camera):
    print("Starting webcam feed...")
    while True:
        frame = camera.get_frame()  # Now returns only the frame (byte string)
        if frame is None:  # Check if frame is None
            print("Failed to capture frame")
            break
        print("Frame captured, size:", len(frame))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    print("Streaming video feed")
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    print("Serving /t endpoint with no data")
    return jsonify([])

@app.route('/recommend', methods=['POST'])
def recommend():
    print("Received /recommend request")
    try:
        data = request.get_json()
        print(f"Request data: {data}")
        type = data.get('type', 'songs')  # Default to songs
        language = data.get('language', '')
        singer = data.get('singer', '')  # Can be singer for songs or genre for movies

        # Get the detected emotion from camera.py
        emotion_index = show_text[0]
        detected_emotion = emotion_dict.get(emotion_index, "Neutral")  # Default to Neutral if invalid
        print(f"Detected emotion: {detected_emotion}")

        # Get song or movie recommendations
        if type == 'songs':
            # Step 1: Load songs from CSV based on emotion
            songs_from_csv = load_emotion_songs(detected_emotion)
            
            if not songs_from_csv:
                print(f"No songs found for emotion: {detected_emotion}")
                return jsonify({
                    "emotion": detected_emotion,
                    "recommendations": [],
                    "message": f"No songs available for {detected_emotion} mood"
                })
            
            # Step 2: Filter songs based on user preferences
            filtered_songs = select_songs_by_preferences(songs_from_csv, language, singer)
            
            # Step 3: Select random songs (up to 10) from the filtered list
            if len(filtered_songs) > 10:
                selected_songs = random.sample(filtered_songs, 10)
            else:
                selected_songs = filtered_songs
            
            print(f"Selected {len(selected_songs)} songs from CSV")
            
            # Step 4: Search each selected song on YouTube
            recommendations = []
            for song in selected_songs:
                youtube_result = search_song_on_youtube(song['name'], song['artist'])
                if youtube_result:
                    recommendations.append(youtube_result)
                else:
                    # If YouTube search fails, create a fallback entry
                    recommendations.append({
                        "title": f"{song['name']} - {song['artist']}",
                        "url": "https://www.youtube.com",
                        "platform": "YouTube (Search manually)",
                        "original_song": song['name'],
                        "original_artist": song['artist']
                    })
            
            # Ensure we have at least some recommendations
            if len(recommendations) < 5:
                # Add more songs if we have fewer than 5
                remaining_songs = [s for s in filtered_songs if s not in selected_songs]
                if remaining_songs:
                    additional_songs = random.sample(remaining_songs, min(5 - len(recommendations), len(remaining_songs)))
                    for song in additional_songs:
                        youtube_result = search_song_on_youtube(song['name'], song['artist'])
                        if youtube_result:
                            recommendations.append(youtube_result)

        elif type == 'movies':
            # For movies, we'll still use the original YouTube search approach
            # since we don't have movie CSV files
            query = f"{detected_emotion.lower()} movie"
            if language:
                query += f" {language}"
            if singer:  # Use as genre
                query += f" {singer}"
            print(f"YouTube query for movies: {query}")

            # Search YouTube for movies
            yt_request = youtube.search().list(
                part="id,snippet",
                q=query,
                type="video",
                maxResults=10,
                videoCategoryId="1"  # Film & Animation category
            )
            response = yt_request.execute()
            recommendations = []
            for item in response["items"]:
                rec = {
                    "title": item["snippet"]["title"],
                    "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "platform": "YouTube"
                }
                recommendations.append(rec)

        print(f"Returning {len(recommendations)} recommendations")
        return jsonify({
            "emotion": detected_emotion,
            "recommendations": recommendations,
            "source": "CSV + YouTube" if type == 'songs' else "YouTube Search"
        })
    except Exception as e:
        print(f"Error in /recommend: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/music')
def music():
    return render_template('music.html')

@app.route('/movie')
def movie():
    return render_template('movie.html')

if __name__ == '__main__':
    print("Starting Flask app...")
    app.debug = True
    app.run()