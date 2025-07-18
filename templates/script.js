document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('music-form');
  const recommendationsDiv = document.getElementById('recommendations');

  form.addEventListener('submit', function(e) {
    e.preventDefault();
    const artist = document.getElementById('artist').value;
    const song = document.getElementById('song').value;
    recommendationsDiv.innerHTML = '<p>Loading recommendations...</p>';

    fetch('/recommend', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ type: 'songs', artist: artist, song: song })
    })
    .then(response => {
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json();
    })
    .then(data => {
      if (!data.recommendations || data.recommendations.length === 0) {
        recommendationsDiv.innerHTML = '<p>No recommendations found.</p>';
        return;
      }
      recommendationsDiv.innerHTML = '';
      data.recommendations.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'album-card';
        itemDiv.innerHTML = `
          <div>
            <span class="type-icon">🎵</span>
            <span class="album-title">${item.title}</span>
          </div>
          <div>
            <a href="${item.url}" target="_blank" class="album-link">${item.platform}</a>
          </div>
        `;
        recommendationsDiv.appendChild(itemDiv);
      });
    })
    .catch(err => {
      recommendationsDiv.innerHTML = '<p class="text-red-500">Error fetching recommendations</p>';
    });
  });
}); 