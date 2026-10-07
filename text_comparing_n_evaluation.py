#VALIDATION & EVALUATION. THEN, CONVERTING IT INTO SCORES OUT OF 10 AFTER COMPARING BOTH THE TEXT.

# Step 1: Install sentence-transformers
!pip install -U sentence-transformers

# Step 2: Import & load model
from sentence_transformers import SentenceTransformer, util

# Load pre-trained model that works well on CPU and Colab
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Define the evaluation function
def evaluate_student_answer(student_keypoints, teacher_keypoints):
    """
    Compares student's answer to teacher's answer using semantic similarity.

    Args:
        student_keypoints (list of str): Bullet points from student's answer.
        teacher_keypoints (list of str): Bullet points from teacher's solution.

    Returns:
        dict: similarity score and feedback
    """
    student_text = " ".join(student_keypoints)
    teacher_text = " ".join(teacher_keypoints)

    student_embedding = model.encode(student_text, convert_to_tensor=True)
    teacher_embedding = model.encode(teacher_text, convert_to_tensor=True)

    similarity = util.cos_sim(student_embedding, teacher_embedding).item()

    if similarity >= 0.85:
        remark = "✅ Excellent: Student's answer is highly relevant and well-aligned."
    elif similarity >= 0.70 and similarity < 80:
        remark = "🟡 Good: Student's answer is mostly relevant with minor missing points."
    elif similarity >= 0.50:
        remark = "⚠️ Fair: Some relevant parts, but several key points are missing or off-topic."
    else:
        remark = "❌ Poor: Student's answer is not contextually relevant to the solution."

    return {
        "similarity_score": round(similarity, 3),
        "feedback": remark
    }

# Step 4: Test the function with a sample example
student_kp = [
    "Solid-state batteries provide higher energy density",
    "Some battery prototypes are fragile outside labs",
    "A power bank is recommended for heavy phone use"
]

teacher_kp = [
    "Chemists are trying to make solid-state batteries commercially viable",
    "These batteries can have 2 to 3 times more energy",
    "They are fragile and don't charge quickly",
    "Reviewers advise power banks for long use"
]

# Get result
result = evaluate_student_answer(student_kp, teacher_kp)
print(result)