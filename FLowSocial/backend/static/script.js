const postsSection = document.getElementById('postsSection');

function createPost() {
    const postContent = document.getElementById('postContent').value;
    if (!postContent) return;

    const post = document.createElement('div');
    post.className = 'post';
    post.innerHTML = `
        <p>${postContent}</p>
        <div class="post-actions">
            <button onclick="likePost(this)">Like</button>
            <span class="like-count">0 likes</span>
            <button onclick="toggleComments(this)">Comments</button>
        </div>
        <div class="comments-section" style="display: none;">
            <input type="text" placeholder="Add a comment" onkeypress="addComment(event, this)">
            <div class="comments"></div>
        </div>
    `;
    postsSection.prepend(post);
    document.getElementById('postContent').value = '';
}

function likePost(button) {
    const likeCount = button.nextElementSibling;
    let count = parseInt(likeCount.innerText);
    likeCount.innerText = `${++count} likes`;
}

function toggleComments(button) {
    const commentsSection = button.parentElement.nextElementSibling;
    commentsSection.style.display = commentsSection.style.display === 'none' ? 'block' : 'none';
}

function addComment(event, input) {
    if (event.key !== 'Enter') return;

    const commentText = input.value;
    if (!commentText) return;

    const comment = document.createElement('div');
    comment.className = 'comment';
    comment.innerText = commentText;

    const commentsDiv = input.nextElementSibling;
    commentsDiv.appendChild(comment);

    input.value = '';
}
