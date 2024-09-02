from flask import Flask, request, jsonify

app = Flask(__name__)

# In-memory storage for posts
posts = []
post_id_counter = 1

@app.route('/posts', methods=['GET'])
def get_posts():
    """Retrieve all posts."""
    return jsonify(posts), 200

@app.route('/posts', methods=['POST'])
def create_post():
    """Create a new post."""
    global post_id_counter
    data = request.json
    post_content = data.get('content')
    if not post_content:
        return jsonify({'error': 'Post content is required'}), 400
    
    post = {
        'id': post_id_counter,
        'content': post_content,
        'likes': 0,
        'comments': []
    }
    posts.append(post)
    post_id_counter += 1
    return jsonify(post), 201

@app.route('/posts/<int:post_id>/like', methods=['POST'])
def like_post(post_id):
    """Like a post."""
    post = next((p for p in posts if p['id'] == post_id), None)
    if post is None:
        return jsonify({'error': 'Post not found'}), 404
    
    post['likes'] += 1
    return jsonify(post), 200

@app.route('/posts/<int:post_id>/comments', methods=['POST'])
def add_comment(post_id):
    """Add a comment to a post."""
    data = request.json
    comment_content = data.get('content')
    if not comment_content:
        return jsonify({'error': 'Comment content is required'}), 400
    
    post = next((p for p in posts if p['id'] == post_id), None)
    if post is None:
        return jsonify({'error': 'Post not found'}), 404
    
    comment = {'content': comment_content}
    post['comments'].append(comment)
    return jsonify(post), 200

if __name__ == '__main__':
    app.run(debug=True)
