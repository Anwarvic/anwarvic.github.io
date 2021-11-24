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
start by listing the files in this project and what each one does.

## Files

The following are all the files found in this repository sorted in alphabetical
order:

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

Now, we have an idea about each single file of this repository. There is one
file that we need to talk about in more details which is `_config.yml`:

## _config.yml

`_config.yml` is a YAML file containing the configuration for the Jekyll server.
You can consider this file as the start-point of the whole project. In this
file, you can define global variables of the whole project. Any file in this
project whether it's an HTML, CSS, JavaScript or even markdown can access these
global variables.

Let's discuss a few of these global variables:

- `title`: The title of the blog.
- `description`: The description of the blog.
- `url`: The url of the deployment.
- `cover`: The image relative path that will be used as a background.
- `baseurl`: The baseurl of the blog. For example, if the `baseurl: 'anwarvic'`,
  this means that the blog will be accessed at http://localhost:4000/anwarvic.
- `google_analytics`: The Google Analytics Tracking ID.
- `disqus_shortname`: The shortname for the disqus plugin.
- `author`: Personal information about the blog owner including his socials.
- `collection_dir`: The directory where all the collections will be found. Mine
  is `my_collections`, so there should be a directory at the root of the project
  with the same name.
- `collections`: A list of all collections in this blog. Each collection is a
  topic; such as "Machine Translation", "Language Modeling", ...etc. Each
  collection has the following properties:
    - `output: true`: This means there will be output for this collection.
    - `permalink`: This is the route of this collection.
    - `title`: The title of the collection.
- `defaults`: All default options can be defined here. Here, I defined the
  default layout for all of my collections; which is `post.html`.
- `destination`: The directory where the project will be built. Mine is `_site`,
  so after starting the server, a new directory called `_site` will be created
  in the root directory.
- `markdown`: The Markdown Flavor used in the project.
- `exclude`: The files that should be excluded and not monitored by the Jekyll
  server.

## Index.html

The index.html file is the main HTMl layout for this project. If you open this
file in this project, you will find the following few lines:
```
---
layout: default
robots: noindex
---
```
This means that 






---
# Features

Starting from this point, I'm going to walk you through the most important
features in this blog and how to customize them:

## Search 


## Create New Collection


## Disqus


## Google Analytics


## MathJax

