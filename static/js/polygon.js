// WebMercator + Esri + imageOverlay on S2 bounds (if present)
const map = L.map('map', { zoomSnap: 0.25, zoomDelta: 0.25 });
L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { attribution: 'Esri' }
).addTo(map);

// نگه‌داشتن رفرنس overlay برای تغییر شفافیت
let overlay = null;

// اسلایدر شفافیت
const slider = document.getElementById('overlayOpacity');
const lbl = document.getElementById('opacityValue');
function applyOpacityFromSlider() {
  if (!overlay) return;
  const v = (parseInt(slider.value, 10) || 0) / 100; // 0..1
  overlay.setOpacity(v);
  if (lbl) lbl.textContent = v.toFixed(2);
}
if (slider) {
  slider.addEventListener('input', applyOpacityFromSlider);
}

fetch('/api/s2_bounds_wgs84')
  .then(r => r.ok ? r.json() : Promise.reject('no_bounds'))
  .then(b => {
    const bounds = [[b.lat_min, b.lon_min],[b.lat_max, b.lon_max]];
    // مقدار اولیه را از اسلایدر بخوان
    const initialOpacity = slider ? (parseInt(slider.value, 10) || 60)/100 : 0.6;

    overlay = L.imageOverlay('/api/output/rgb_quicklook.png', bounds, { opacity: initialOpacity }).addTo(map);
    if (lbl) lbl.textContent = initialOpacity.toFixed(2);

    map.fitBounds(bounds);
    initDraw(map);
  })
  .catch(() => {
    map.setView([29.0, 52.0], 12); // fallback view
    initDraw(map);
  });

function initDraw(map){
  const drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  fetch('/api/output/polygons.geojson')
    .then(r=> r.ok ? r.json() : null)
    .then(g=>{
      if(g){
        L.geoJson(g, {
          onEachFeature: (_, layer)=> drawnItems.addLayer(layer),
          style: f => ({
            color: (f.properties && f.properties.color) ? f.properties.color : '#00ff00',
            weight: 2
          })
        });
      }
    });

  const drawControl = new L.Control.Draw({
    draw: { polygon: { shapeOptions: { color: '#00ff00', weight:2 } }, polyline:false, rectangle:false, circle:false, marker:false, circlemarker:false },
    edit: { featureGroup: drawnItems }
  });
  map.addControl(drawControl);

  map.on(L.Draw.Event.CREATED, e => drawnItems.addLayer(e.layer));

  document.getElementById('savePolygonsBtn').onclick = async () => {
    const fc = { type:'FeatureCollection', features:[] };
    drawnItems.eachLayer(layer=>{
      try{
        const gj = layer.toGeoJSON();
        gj.properties = gj.properties || {};
        gj.properties.color = (layer.options && layer.options.color) ? layer.options.color : '#00ff00';
        fc.features.push(gj);
      }catch(err){ console.warn(err); }
    });
    const r = await fetch('/api/save_polygons', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(fc)
    });
    alert(r.ok ? 'ذخیره شد' : 'خطا در ذخیره');
  };
}