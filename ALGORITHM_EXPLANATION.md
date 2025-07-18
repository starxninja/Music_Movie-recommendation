# CSV-Based Song Recommendation Algorithm

## Overview
Instead of using generic emotion-based YouTube searches, this algorithm first selects songs from your Excel/CSV files based on detected emotion, then searches those specific songs on YouTube.

## Algorithm Flow

### 1. **Emotion Detection** (from camera.py)
```
Webcam → Face Detection → CNN Model → Emotion Classification
```
- Detects one of 7 emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
- Stores emotion index in global variable `show_text[0]`

### 2. **CSV Song Selection** (new algorithm)
```
Detected Emotion → Load CSV File → Filter by Preferences → Random Selection
```

#### Step 1: Load Songs from CSV
```python
def load_emotion_songs(emotion):
    csv_file = f"songs/{emotion.lower()}.csv"
    df = pd.read_csv(csv_file)
    # Convert to list of song dictionaries
```

**Example for "Happy" emotion:**
- Loads `songs/happy.csv`
- Contains 82 songs like:
  - "Leave The Door Open" by Bruno Mars
  - "Dynamite" by BTS
  - "Levitating" by Dua Lipa
  - etc.

#### Step 2: Filter by User Preferences
```python
def select_songs_by_preferences(songs, language="", artist=""):
    if artist:
        filtered_songs = [song for song in songs 
                         if artist.lower() in song['artist'].lower()]
```

**Example filtering:**
- Original: 82 happy songs
- Filter by "Bruno Mars" → 4 songs
- Filter by "BTS" → 1 song

#### Step 3: Random Selection
```python
if len(filtered_songs) > 10:
    selected_songs = random.sample(filtered_songs, 10)
else:
    selected_songs = filtered_songs
```

### 3. **YouTube Search** (for selected songs)
```
Selected Song → YouTube API Search → Video URL
```

#### Step 4: Search Each Song on YouTube
```python
def search_song_on_youtube(song_name, artist_name):
    query = f"{song_name} {artist_name} official"
    yt_request = youtube.search().list(
        q=query,
        type="video",
        maxResults=1,
        videoCategoryId="10"  # Music category
    )
```

**Example search:**
- Song: "Leave The Door Open"
- Artist: "Bruno Mars"
- Query: "Leave The Door Open Bruno Mars official"
- Result: Direct YouTube video link

## Complete Example

### Scenario: User is Happy, wants Bruno Mars songs

1. **Emotion Detection**: Camera detects "Happy" emotion
2. **CSV Loading**: Loads `songs/happy.csv` (82 songs)
3. **Filtering**: Filters to songs by Bruno Mars (4 songs)
4. **Selection**: Randomly selects 4 songs
5. **YouTube Search**: Searches each song individually
6. **Results**: Returns 4 direct YouTube links

### Before vs After

**OLD METHOD (Generic Search):**
```
Query: "happy song bruno mars"
Results: Random YouTube videos matching query
```

**NEW METHOD (CSV + YouTube):**
```
1. Load: 82 happy songs from CSV
2. Filter: 4 Bruno Mars songs
3. Search: Each song individually on YouTube
4. Results: 4 specific song videos
```

## Advantages

1. **Precise Selection**: Uses curated song lists instead of random YouTube results
2. **Better Quality**: Songs are pre-selected for each emotion
3. **Artist Filtering**: Can filter by specific artists from your database
4. **Consistent Results**: Same emotion always gives similar song pool
5. **Fallback Support**: If YouTube search fails, still shows song info

## File Structure

```
songs/
├── angry.csv      (32 songs)
├── disgusted.csv  (102 songs)
├── fearful.csv    (32 songs)
├── happy.csv      (82 songs)
├── neutral.csv    (102 songs)
├── sad.csv        (102 songs)
└── surprised.csv  (52 songs)
```

Each CSV contains:
- Name: Song title
- Album: Album name
- Artist: Artist name

## Usage

1. **Start the application**: `python app.py`
2. **Open web interface**: Navigate to music page
3. **Camera detects emotion**: Automatically detects your mood
4. **Add preferences**: Optional artist/language filters
5. **Get recommendations**: Algorithm selects songs from CSV, searches YouTube
6. **Click links**: Direct to specific song videos

## Testing

Run the test script to see the algorithm in action:
```bash
python test_algorithm.py
```

This will demonstrate:
- Loading songs from different emotion CSV files
- Filtering by artist preferences
- Simulating YouTube searches
- Showing final recommendations 