import pickledb

Posts = pickledb.load("../posts.db", True)


def insert_post(post):
    Posts.set(post['id'], post)


def insert_posts(posts):
    for post in posts:
        insert_post(post)
