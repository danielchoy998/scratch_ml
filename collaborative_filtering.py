# Item-based CF
import numpy as np

items = ["items1", "items2", "items3", "items4"]

user_1 = np.array([5, 3, 4, 4])
user_2 = np.array([3, 1, 3, 3])
user_3 = np.array([4, 2, 4, 1])
user_4 = np.array([4, 3, 3, 5])
user_5 = np.array([0, 5, 4, 0])

users_rating_matrix = np.array([user_1, user_2, user_3, user_4, user_5])

item_rating_matrix = np.transpose(users_rating_matrix)


def compute_similarity(item_1, item_2):
    dot_product = np.dot(item_1, item_2)
    magnitude_1 = np.linalg.norm(item_1)
    magnitude_2 = np.linalg.norm(item_2)

    return dot_product / (magnitude_1 * magnitude_2)


def create_matrix(items, users_rating_matrix):
    similarity_matrix = np.zeros((len(items), len(items)))
    for i in range(len(items)):
        for j in range(len(items)):
            if i == j:
                similarity_matrix[i, j] = 1
            else:
                similarity_matrix[i, j] = compute_similarity(item_rating_matrix[i], item_rating_matrix[j])

    return similarity_matrix


def predict_rating(new_user_rating, similarity_matrix):
    predicted_ratings = new_user_rating.copy()
    for i in range(len(items)):
        if predicted_ratings[i] > 0:
            continue
        
        rated_idx = np.where(new_user_rating>0)[0]
        if len(rated_idx) == 0:
            continue

        similar_scores = similarity_matrix[i, rated_idx] # similarity between the predicting item and the rated items
        ratings = new_user_rating[rated_idx] # ratings of the rated items

        predicted_ratings[i] = np.sum(similar_scores * ratings) / np.sum(np.abs(similar_scores))
    return predicted_ratings
       
    
similarity_matrix = create_matrix(items, users_rating_matrix)
print(similarity_matrix)

new_user_rating = np.array([4, 5, 0, 0])
print(predict_rating(new_user_rating, similarity_matrix))

            
            
