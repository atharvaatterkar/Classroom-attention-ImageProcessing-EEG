// Handle sidebar menu link clicks
const sideLinks = document.querySelectorAll('.sidebar .side-menu li a:not(.logout)');

sideLinks.forEach(item => {
    const li = item.parentElement;
    item.addEventListener('click', () => {
        sideLinks.forEach(i => {
            i.parentElement.classList.remove('active');
        });
        li.classList.add('active');
    });
});

// Toggle sidebar visibility on menu bar click
const menuBar = document.querySelector('.content nav .bx.bx-menu');
const sideBar = document.querySelector('.sidebar');

menuBar.addEventListener('click', () => {
    sideBar.classList.toggle('close');
});

// Handle search button functionality for mobile view
const searchBtn = document.querySelector('.content nav form .form-input button');
const searchBtnIcon = document.querySelector('.content nav form .form-input button .bx');
const searchForm = document.querySelector('.content nav form');

searchBtn.addEventListener('click', (e) => {
    if (window.innerWidth < 576) {
        e.preventDefault();
        searchForm.classList.toggle('show');
        searchBtnIcon.classList.toggle('bx-search');
        searchBtnIcon.classList.toggle('bx-x');
    }
});

// Handle window resize for responsive design
window.addEventListener('resize', () => {
    if (window.innerWidth < 768) {
        sideBar.classList.add('close');
    } else {
        sideBar.classList.remove('close');
    }
    if (window.innerWidth > 576) {
        searchBtnIcon.classList.replace('bx-x', 'bx-search');
        searchForm.classList.remove('show');
    }
});

// Handle theme toggling
const toggler = document.getElementById('theme-toggle');

toggler.addEventListener('change', function() {
    if (this.checked) {
        document.body.classList.add('dark');
        document.querySelector('.profile>img').style.filter = 'invert(1)';
        document.querySelector('#dash').style.color = 'white';
    } else {
        document.body.classList.remove('dark');
        document.querySelector('#dash').style.color = 'black';
        document.querySelector('.profile>img').style.filter = 'invert(0)';
    }
});