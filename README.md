# [anwarvic.github.io](https://anwarvic.github.io)

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

- `css/`: directory containing `main.css` file which contains my preferred
  styles. This file overrides the properties found in `_scss` directory.

- `_drafts/`: directory for drafts of your posts. This directory is excluded
  by default in Jekyll.

- `images/`: directory containing images found only the cover of the blog.
  Posts images can be found in `my_collections` directory.

- `_includes/`: directory containing relatively small HTML layout files that
  will be included by the HTML layout files defined in the `layout` directory.
  - `cover.html`: The blog cover! The part where the name, socials, and
    collection icons are found.
  - `disqus.html`: For the disqus plugin, gonna talk about his [later]().
  - `footer.html`: The footer for all pages in the blog.
  - `head.html`: The header for all pages in the blog.
  - `socials.html`: The HTML page for all social icons found on the cover.

- `js/`: directory containing all JavaScript scripts in this blog.
  - `jquery.v3.3.1.min.js`: JQuery v3.3.1 (included so I can work offline).
  - `main.js`: User-defined functions.
  - `search.json`: JSON file that creates blog database for the search
    functionality. Gonna talk about in more detail later.
  - `simple-blog-search.min.js`:
  [Simple Blog Search](https://github.com/SeraphRoy/SimpleBlogSearch)
  plugin used for the search functionality.

- `_layouts/`: directory for the main HTML layouts used in the blog.
  - `default.html`: The main (default) HTML layout for the blog.
  - `named_collection.html`: The HTML layout for enlisting all articles found
    in a certain collection.
  - `post.html`: The HTML layout for the article/post.

- `my_collections/`: directory containing all articles I wrote for my blog. All
  files in this directory are in Markdown format. Any images included in any
  article can be found here as well.

- `_sass/`: directory containing all SCSS files.
  - `animate.scss`: defines simple animation used in the blog; like the
  collapse or bounce down.
  - `monokai.scss`: defines the style of the inline code in posts.
  - `tables.scss`: defines the style of tables in posts.
  - `uno.scss`: defines the main style for the blog.

- `404.md`: file for not-found pages.
- `_config.yml`: YAML configuration file for Jekyll.
- `Gemfile`:
- `Gemfile.lock`:
- `index.html`: The home page.
- `language-modeling.md`: The main page for `/language-modeling` route.
- `machine-translation.md`: The main page for `/machine-translation` route.
- `multilingual-nmts.md`: The main page for `/multilingual-nmts` route.
- `search.html`: The main page for the `/search` route.
- `sitemap.xml`: 
- `speech-recognition.html`: The main page for the `/speech-recognition` route.
- `word-embedding.html`: The main page for the `/word-embedding` route.











# Features

Starting from this point, I'm going to walk you through the most important
features in this blog and how to customize them:

## Search 


## Create New Collection


## Disqus


## Google Analytics


## MathJax

