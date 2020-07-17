import path from 'path';
import fs from 'fs';
import { siftMatches } from './lib/3dReconstruction/findMatches';
import { Matrix } from 'ml-matrix';
import * as cv from 'opencv4nodejs';
import { eightPointAlgorithm } from './lib/3dReconstruction/eightPointAlgorithm';
import { ndMat, IMatchPoint, cvMat, IPoint } from './@types';
import { ransac } from './lib/Optimization/ransac';
import { getRandomInt, normalizeVertexList } from './lib/utils';
import { matrix2ndMat, readCameraProjectionMatrixFromFile } from './lib/utils';
import { passiveTriangulation } from './lib/3dReconstruction/passiveTriangulation';
import { computeEpilines, searchMatchInEpiline, getPixelWindow } from './lib/3dReconstruction/epilines';

// Draw epilines for debug purposes
const drawLines = (lines: Array<IPoint>, img: cvMat) => {
  const c = img.cols;
  for (let i = 0; i < lines.length; i++) {
    const color = new cv.Vec3(getRandomInt(0, 255), getRandomInt(0, 255), getRandomInt(0, 255));
    let [x0, y0] = [0, -lines[i][2] / lines[i][1]].map(el => Math.round(el));
    let [x1, y1] = [c, -(lines[i][2] + lines[i][0] * c) / lines[i][1]].map(el => Math.round(el));
    img.drawLine(new cv.Point2(x0, y0), new cv.Point2(x1, y1), color, 1);
  }
};

// Error function to ransac
const calculateSquareError = (sample: IMatchPoint, F: ndMat) => {
  const [x, y] = sample.p1, [xl, yl] = sample.p2;
  const xTmp = new Matrix([[x], [y], [1]]);
  let xlTmp = new Matrix([[xl, yl, 1]]);
  let M = xlTmp.mmul(new Matrix(F)).mmul(xTmp);
  let m = matrix2ndMat(M), sum = 0;
  for (let i = 0; i < m.length; i++) {
    for (let j = 0; j < m[i].length; j++) {
      sum += (m[i][j] * m[i][j]);
    }
  }
  return sum;
};

// * Find and draw epilines for two images (Debug purposes)

// Read the two images
const img1cv = cv.imread(path.resolve('.', 'datasets', 'dino', 'dino0001.png'));
const img2cv = cv.imread(path.resolve('.', 'datasets', 'dino', 'dino0002.png'));

// find matches between the two images
const customMatches = siftMatches(img1cv, img2cv, 5000);

// Computing fundamental matrix
let customF = ransac(customMatches, 8, eightPointAlgorithm, calculateSquareError, 0.1, 0.98, 10);

const linesOnLeft = computeEpilines(customMatches.map(el => el.p2), customF, true);
const linesOnRight = computeEpilines(customMatches.map(el => el.p1), customF);

// Draw lines and show images
drawLines(linesOnLeft, img1cv);
drawLines(linesOnRight, img2cv);

cv.imshowWait('f1.png', img1cv);
cv.imshowWait('f2.png', img2cv);


// * 3d reconstruction from multiview

const pad = (num: number, size: number = 4) => {
  let s = `${num}`;
  while (s.length < size) {
    s = '0' + s;
  }
  return s;
};

let vertexList: ndMat = [];
for (let i = 1; i < 363; i += 2) {
  console.log(i);
  const img1cvMat = cv.imread(path.resolve('.', 'datasets', 'dino', `dino${pad(i)}.png`));
  const img2cvMat = cv.imread(path.resolve('.', 'datasets', 'dino', `dino${pad(i + 1)}.png`));

  // Convert to ndMat
  const img2 = img2cvMat.getDataAsArray();
  const img1 = img1cvMat.getDataAsArray();

  // Read the projections matrices
  const P1 = readCameraProjectionMatrixFromFile(path.resolve('.', 'datasets', 'dino', 'dino_par.txt'), `dino${pad(i)}.png`);
  const P2 = readCameraProjectionMatrixFromFile(path.resolve('.', 'datasets', 'dino', 'dino_par.txt'), `dino${pad(i + 1)}.png`);

  const matches = siftMatches(img1cvMat, img2cvMat, 5000);
  const F = ransac(matches, 8, eightPointAlgorithm, calculateSquareError, 0.1, 0.98, 10);

  // Recover depth for sift matches
  matches.forEach(match => {
    const pt = [Math.round(match.p1[0]), Math.round(match.p1[1])];
    const w = getPixelWindow(img1, pt, 1);
    if (w) {
      const epiline = computeEpilines([pt], F)[0];
      const matchPosition = searchMatchInEpiline(img2, w, epiline, 'SSD');
      vertexList.push(passiveTriangulation(pt, matchPosition, P1, P2));
    }
  });

  // Recover depth by passive triangulation for all pixels
  /*for (let i = 0; i < img1.length; i++) {
    for (let j = 0; j < img1[i].length; j++) {
      const w = getPixelWindow(img1, [i, j], 5);
      if (w) {
        const epiline = computeEpilines([[i, j]], F)[0];
        const matchPosition = searchMatchInEpiline(img2, w, epiline, 'SSD');
        vertexList.push(passiveTriangulation([i, j], matchPosition, P1, P2));
      }
    }
    console.log(i);
  }*/

}

fs.writeFileSync('vertexPoints.txt', JSON.stringify(vertexList));
// fs.writeFileSync('vertexPoints.txt', JSON.stringify(normalizeVertexList(vertexList))); // Normalized vertex list

