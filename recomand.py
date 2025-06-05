import pandas as pd
import numpy as np

# 1. Sample dataset
data = {
    'User': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
    'Movie': ['Inception', 'Titanic', 'Avatar', 'Inception', 'Titanic',
              'Avatar', 'Titanic', 'Inception', 'Avatar', 'Titanic'],
    'Rating': [5, 4, 4, 5, 3, 2, 5, 4, 5, 3]
}

df = pd.DataFrame(data)

# 2. Create user-movie matrix
rating_matrix = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

# 3. Cosine similarity function
from numpy.linalg import norm
def cosine_similarity(u, v):
    return np.dot(u, v) / (norm(u) * norm(v) + 1e-10)

# 4. Target user
target_user = 'C'
similarities = {}

for user in rating_matrix.index:
    if user != target_user:
        sim = cosine_similarity(rating_matrix.loc[target_user], rating_matrix.loc[user])
        similarities[user] = sim

# 5. Most similar user
most_similar_user = max(similarities, key=similarities.get)
print(f"Most similar user to {target_user}: {most_similar_user} (similarity: {similarities[most_similar_user]:.2f})")

# 6. Recommend movies
recommendations = []

for movie in rating_matrix.columns:
    user_rating = rating_matrix.loc[target_user, movie]
    similar_user_rating = rating_matrix.loc[most_similar_user, movie]

    if user_rating == 0 and similar_user_rating > 3:
        recommendations.append((movie, similar_user_rating))

# 7. Show recommendations
print(f"\nRecommended movies for {target_user}:")
if recommendations:
    for movie, rating in recommendations:
        print(f"- {movie} (similar user rated it {rating})")
else:
    print("No recommendations found.")
