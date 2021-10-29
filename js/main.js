---
  layout: null
sitemap:
exclude: 'yes'
---


$(document).ready(function () {
  {% if site.disable_landing_page != true %}
  // $('a.blog-button').click(function (e) {
  //   if ($('.panel-cover').hasClass('panel-cover--collapsed')){
  //     return
  //   }
  //   currentWidth = $('.panel-cover').width()
  //   if (currentWidth < 960) {
  //     $('.panel-cover').addClass('panel-cover--collapsed')
  //     // $('.content-wrapper').addClass('animated slideInRight')
  //   } else {
  //     $('.panel-cover').css('max-width', currentWidth)
  //     $('.panel-cover').animate({ 'max-width': '530px', 'width': '40%' }, 400, swing = 'swing', function () { })
  //   }
  // })

  // if (window.location.hash && window.location.hash == 'machine-translation') {
  //   $('.panel-cover').addClass('panel-cover--collapsed')
  // }

  if (window.location.pathname !== '{{ site.baseurl }}/' && window.location.pathname !== '{{ site.baseurl }}/index.html') {
    $('.panel-cover').addClass('panel-cover--collapsed')
  }
  {% endif %}

  $('.btn-mobile-menu').click(function () {
    $('.navigation-wrapper').toggleClass('visible animated bounceInDown')
    $('.btn-mobile-menu__icon').toggleClass('icon-list icon-x-circle animated fadeIn')
  })

  $('.navigation-wrapper .blog-button').click(function () {
    $('.navigation-wrapper').toggleClass('visible')
    $('.btn-mobile-menu__icon').toggleClass('icon-list icon-x-circle animated fadeIn')
  })

})

// only zero or one button should be active
window.activeButtons = [];
const maxwidth = 2000;

function collapse() {
  currentWidth = $('.panel-cover').width()
  $('.panel-cover').css('max-width', currentWidth)
  $('.panel-cover').animate({ 'max-width': '530px', 'width': '40%' }, 500, swing = 'swing', function () { })
  $('.panel-cover').addClass('panel-cover--collapsed')
}

function expand() {
  $('.panel-cover').css('max-width', maxwidth)
  $('.panel-cover').animate({ 'max-width': maxwidth, 'width': '100%' }, 500, swing = 'swing', function () { })
  $('.panel-cover').removeClass('panel-cover--collapsed')
}

function generateContent(id){
  const txt = `
  <ol class="post-list">
  <h1>{{ page.url }}</h1>
  {% for post in site.machine-translation%}
    <li>
      <h2 class="post-list__post-title post-title"><a href="{{ site.baseurl }}{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a></h2>
      <p class="excerpt">{{ post.excerpt | strip_html }}&hellip;</p>
      <div class="post-list__meta">
          <time datetime="{{ post.date | date: '%Y-%m-%d %H:%M' }}" class="post-list__meta--date date">{{ post.date | date: "%-d %b %Y" }}</time>
          {% if post.tags.size > 0 %}
          &#8226; <span class="post-meta__tags">on {% for tag in post.tags %}<a href="{{ site.baseurl }}/tags/#{{ tag }}">{{ tag }}</a>{% if forloop.last == false %}, {% endif %}{% endfor %}</span>
          {% endif %}
      </div>
      <hr class="post-list__divider">
    </li>
  {% endfor %}
  </ol>
  `;
  console.log(txt);
  $("#list-items").html(txt);
}

function activateButton(id) {
  const elem = $('#'+id);
  // deactivate
  if (window.activeButtons.includes(id)) {
    elem.removeClass('clicked');
    expand();
  }
  // activate
  else {
    if (window.activeButtons.length > 0) {
      $('#'+window.activeButtons.pop()).removeClass("clicked");
    }
    window.activeButtons.push(id);
    elem.addClass('clicked');
    collapse();
    // generateContent(id);
  }
}
