// static/main.js

// Placeholder for local model and training data
let localModel = { trained: false };  // Initialize model state
let interactions = [];

function likePost(postId) {
    fetch('/like_post', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user: 'user_1', post_id: postId })
    }).then(response => response.json())
      .then(data => {
        document.getElementById("status_" + postId).innerText = "Post liked!";
        interactions.push({ postId: postId, action: 'like' });
    }).catch(error => console.error('Error:', error));
}

function showCommentBox(postId) {
    document.getElementById("comment_box_" + postId).style.display = 'block';
}

function commentPost(postId) {
    const commentText = document.getElementById("comment_text_" + postId).value;
    fetch('/comment_post', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user: 'user_1', post_id: postId, comment: commentText })
    }).then(response => response.json())
      .then(data => {
        document.getElementById("status_" + postId).innerText = "Comment added!";
        interactions.push({ postId: postId, action: 'comment', comment: commentText });
    }).catch(error => console.error('Error:', error));
}

function trainLocally() {
    // Simulate local training based on interactions
    if (interactions.length > 0) {
        localModel.trained = true;
        document.getElementById("status").innerText = "Model trained locally with interactions!";
    } else {
        document.getElementById("status").innerText = "No interactions to train on.";
    }
}

function sendUpdates() {
    if (!localModel.trained) {
        document.getElementById("status").innerText = "Train the model locally first!";
        return;
    }

    // Simulate sending model updates to the server
    fetch('/send_updates', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(localModel)
    }).then(response => response.json())
      .then(data => {
        document.getElementById("status").innerText = "Updates sent to server!";
    }).catch(error => console.error('Error:', error));
}
