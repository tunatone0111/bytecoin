from flask import Blueprint

posts = Blueprint("posts", __name__)

@posts.route("/", methods=['GET'])
def get_posts():
    pass
