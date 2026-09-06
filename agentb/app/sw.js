// Pocket Mnemo — offline SHELL only. Memory never lives on the phone (v0).
// Cache the door; every /chat call goes to the engine, always.
var CACHE = 'pocket-mnemo-shell-v1';
var SHELL = ['/app/', '/app/manifest.webmanifest', '/app/icon.svg'];
self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(CACHE).then(function (c) { return c.addAll(SHELL); }).then(function () { return self.skipWaiting(); }));
});
self.addEventListener('activate', function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
self.addEventListener('fetch', function (e) {
  var url = new URL(e.request.url);
  if (e.request.method !== 'GET' || !url.pathname.startsWith('/app/')) return; // /chat: network, no cache, ever
  // network first (fresh shell after a deploy), cache when the engine is unreachable
  e.respondWith(fetch(e.request).then(function (r) {
    var copy = r.clone(); caches.open(CACHE).then(function (c) { c.put(e.request, copy); }); return r;
  }).catch(function () { return caches.match(e.request); }));
});
