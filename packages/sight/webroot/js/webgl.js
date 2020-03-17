/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Creates a 3JS object to render a coordinate frame
function CreateIsaacAxes(size, width) {
  let axes = new THREE.AxesHelper(size);
  axes.material.linewidth = width;
  return axes;
}

// Create a 3JS object to render an occupancy map as a flat textured map
function CreateFlatFloorMap(image, pixelsize) {
  let scale = pixelsize;
  // a large quad on the ground
  let floor_geo = new THREE.Geometry();
  floor_geo.vertices.push(new THREE.Vector3(0,0,0));
  floor_geo.vertices.push(new THREE.Vector3(0,scale*image.width,0));
  floor_geo.vertices.push(new THREE.Vector3(scale*image.height, scale*image.width,0));
  floor_geo.vertices.push(new THREE.Vector3(scale*image.height,0,0));
  floor_geo.faces.push(new THREE.Face3(0, 1, 2, new THREE.Vector3(0,1,0), new THREE.Color(0xffffff), 0));
  floor_geo.faces.push(new THREE.Face3(0, 2, 3, new THREE.Vector3(0,1,0), new THREE.Color(0xffffff), 0));
  floor_geo.faceVertexUvs[0] = [];
  floor_geo.faceVertexUvs[0].push([
    new THREE.Vector2(0,1),
    new THREE.Vector2(1,1),
    new THREE.Vector2(1,0)
  ]);
  floor_geo.faceVertexUvs[0].push([
    new THREE.Vector2(0,1),
    new THREE.Vector2(1,0),
    new THREE.Vector2(0,0)
  ]);
  floor_geo.computeVertexNormals();
  let floor_mat = new THREE.MeshLambertMaterial({ color: 0x999999, side: THREE.DoubleSide });
  let texture = new THREE.Texture();
  texture.image = image;
  texture.format = THREE.RGBAFormat;
  texture.needsUpdate = true;
  floor_mat.map = texture;  //texloader.load(texture_name);
  floor_mat.map.magFilter = THREE.NearestFilter;
  floor_mat.map.minFilter = THREE.NearestFilter;
  let plane = new THREE.Mesh(floor_geo, floor_mat);
  plane.receiveShadow = true;
  plane.castShadow = false;
  return plane;
}

// Create a 3JS object to render an occupancy map as an extruded height map
function CreateExtrudedFloorMap(image, pixelsize, height, limit = 100000) {
  let blockunits = pixelsize;

  let matrix = new THREE.Matrix4();
  let pxGeometry = new THREE.PlaneBufferGeometry(blockunits, height);
  pxGeometry.attributes.uv.array[ 1 ] = 0.5;
  pxGeometry.attributes.uv.array[ 3 ] = 0.5;
  pxGeometry.rotateX(-Math.PI / 2 );
  pxGeometry.translate(0.5*blockunits, blockunits, 0);

  let nxGeometry = new THREE.PlaneBufferGeometry(blockunits, height);
  nxGeometry.attributes.uv.array[ 1 ] = 0.5;
  nxGeometry.attributes.uv.array[ 3 ] = 0.5;
  nxGeometry.rotateX(Math.PI / 2);
  nxGeometry.translate(0.5*blockunits, 0, 0);

  let pyGeometry = new THREE.PlaneBufferGeometry(blockunits, blockunits);
  pyGeometry.attributes.uv.array[ 5 ] = 0.5;
  pyGeometry.attributes.uv.array[ 7 ] = 0.5;
  pyGeometry.rotateZ(-Math.PI / 2);
  pyGeometry.translate(0.5*blockunits, 0.5*blockunits, 0.5 * height);

  let pzGeometry = new THREE.PlaneBufferGeometry(height, blockunits);
  pzGeometry.attributes.uv.array[ 1 ] = 0.5;
  pzGeometry.attributes.uv.array[ 3 ] = 0.5;
  pzGeometry.rotateY(Math.PI / 2);
  pzGeometry.translate(blockunits, 0.5*blockunits, 0);
  let nzGeometry = new THREE.PlaneBufferGeometry(height, blockunits);
  nzGeometry.attributes.uv.array[ 1 ] = 0.5;
  nzGeometry.attributes.uv.array[ 3 ] = 0.5;
  nzGeometry.rotateY(Math.PI / 2);
  nzGeometry.translate(0, 0.5*blockunits, 0);

  let tmpGeometry = new THREE.Geometry();
  let pxTmpGeometry = new THREE.Geometry().fromBufferGeometry( pxGeometry );
  let nxTmpGeometry = new THREE.Geometry().fromBufferGeometry( nxGeometry );
  let pyTmpGeometry = new THREE.Geometry().fromBufferGeometry( pyGeometry );
  let pzTmpGeometry = new THREE.Geometry().fromBufferGeometry( pzGeometry );
  let nzTmpGeometry = new THREE.Geometry().fromBufferGeometry( nzGeometry );

  // It seems that chrome sometimes have issue calling getImageData on big canvas, therefore we
  // split the image in smaller blocks.
  const max_dims = 256;

  function getImageData(image, row, col) {
    let canvas = document.createElement('canvas');
    canvas.width = Math.min(max_dims, image.width - col);
    canvas.height = Math.min(max_dims, image.height - row);
    let context = canvas.getContext('2d');
    context.drawImage(image, -col, -row);
    return context.getImageData(0, 0, canvas.width, canvas.height);
  }

  function getY(x, z, imgdata) {
    const q = imgdata.data[4*(x + imgdata.width*z)];
    return (q < 120) ? 0 : 1;
  }

  for (let block_z = 0; block_z * max_dims < image.height; block_z++) {
    for (let block_x = 0; block_x * max_dims < image.width; block_x++) {
      const start_x = block_x * max_dims;
      const start_z = block_z * max_dims;
      let imgdata = getImageData(image, start_z, start_x);
      const worldDepth = imgdata.height;
      const worldWidth = imgdata.width;

      for (let z=0; z<worldDepth; z+=1) {
        for (let x=0; x<worldWidth; x+=1) {
          const h = getY(x, z, imgdata);
          if (h != 0) continue;
          if (limit == 0) break;
          limit--;
          matrix.makeTranslation((start_z + z)*blockunits, (start_x + x)*blockunits, height / 2);
          tmpGeometry.merge(pyTmpGeometry, matrix);
          if (x === 0 || getY(x - 1, z, imgdata) !== h) {
            tmpGeometry.merge(nxTmpGeometry, matrix);
          }
          if (x === worldWidth - 1 || getY(x + 1, z, imgdata) != h) {
            tmpGeometry.merge(pxTmpGeometry, matrix);
          }
          if (z === worldDepth - 1 || getY(x, z + 1, imgdata) != h) {
            tmpGeometry.merge(pzTmpGeometry, matrix);
          }
          if (z === 0 || getY(x, z - 1, imgdata) != h) {
            tmpGeometry.merge(nzTmpGeometry, matrix);
          }
        }
      }
    }
  }

  let geometry = new THREE.BufferGeometry().fromGeometry(tmpGeometry);
  geometry.computeBoundingSphere();

  let mat = new THREE.MeshLambertMaterial({color: 0x333333, side: THREE.DoubleSide});

  let mesh = new THREE.Mesh(geometry, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

// Creates visualization for a map
function CreateMap(image, pixelsize, height = 200, floor_z = 0) {
  let floor = new THREE.Group();
  // flat floor map
  floor.add(CreateFlatFloorMap(image, pixelsize));
  // extruded floor map
  floor.add(CreateExtrudedFloorMap(image, pixelsize, height));
  floor.translateZ(floor_z);
  return floor;
}
