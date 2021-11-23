# anwarvic.github.io

This is the source code for my blog hosted in the following
url: https://anwarvic.github.io. This source code was adapted from the
[Jekyll-Uno](https://github.com/joshgerdes/jekyll-uno) Theme.

<div align="center">
  <img src="/images/assets/peek_web.gif" width=750>
  <img src="/images/assets/peek_mobile.gif" height=500>
</div>

## Dependencies

The following are the prerequisites that need to be installed to be able to 
build the website offline:

- Install ruby; this is the command on Ubuntu
  ```
  sudo apt-get install ruby-full build-essential zlib1g-dev
  ```
- Install bundler
  ```
  gem install bundler
  ```

## Run it offline

Now all dependencies are installed, you can follow these steps to be able to
build the blog offline on your machine:

- clone the repository from GitHub:
  ```
  git clone https://github.com/Anwarvic/anwarvic.github.io.git
  ```
- Install Ruby gems:
  ```
  bundle install
  ```
- Now, we can start Jekyll server:
  ```
  bundle exec jekyll serve --watch
  ```
- You can access the blog via the following link: http://localhost:4000


---

# How It works

In this part, I will try to explain how things work in this repository. Let's
start by listing the files in this project and what each one does:

- `css`: directory containing `main.css` file which contains my preferred
  styles. This file overrides the properties found in `_scss` directory.
- `images`: directory containing images found only the cover of the blog.
  Posts images can be found in `my_collections` directory.
- `_includes`: directory containing HTML layout files.
  - `disqus.html`: For the disqus plugin, gonna talk about his [later]().
  - `footer.html`: The footer HTML layout for all pages in the blog.
  - `head.html`: The 

- ``



# Features

Starting from this point, I'm going to walk you through the most important
features in this blog and how to customize them:

## Search 


## Create New Collection

## Disqus


## Google Analytics