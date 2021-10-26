---
  layout: null
sitemap:
exclude: 'yes'
---


$(document).ready(function () {
  // {% if site.disable_landing_page != true %}
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

  // if (window.location.hash && window.location.hash == '#blog') {
  //   $('.panel-cover').addClass('panel-cover--collapsed')
  // }

  // if (window.location.pathname !== '{{ site.baseurl }}/' && window.location.pathname !== '{{ site.baseurl }}/index.html') {
  //   $('.panel-cover').addClass('panel-cover--collapsed')
  // }
  // {% endif %}

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

function activateButton(id) {
  const elem = $('#'+id);
  // deactivate
  if (window.activeButtons.includes(id)) {
    elem.removeClass('clicked');
    window.activeButtons.pop();
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
  }
}
