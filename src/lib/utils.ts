import Matrix from "ml-matrix";
import { ndMat } from "../@types";
import fs from 'fs';
import path from 'path';

// Convert Matrix instance to NODE.Js array
export const matrix2ndMat = (m: Matrix): ndMat => {
  let resp = [];
  for (let i = 0; i < m.rows; i++) {
    resp[i] = m.getRow(i);
  }
  return resp;
};

export const getRandomInt = (min: number, max: number): number => {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

export const readCameraProjectionMatrixFromFile = (pathToParameters: string, filename: string) => {
  const fileLines = fs.readFileSync(pathToParameters, 'utf-8').split('\n');
  let K = [] as ndMat;
  let R = [] as ndMat;

  fileLines.forEach((line: string) => {
    const parts = line.split(' ');
    if (parts[0] === filename) {
      for (let i = 0; i < 3; i++) {
        K[i] = [];
        for (let j = 0; j < 3; j++) {
          K[i][j] = Number(parts[(i * 3 + j) + 1]);
        }
      }
      for (let i = 0; i < 3; i++) {
        R[i] = [];
        for (let j = 0; j < 3; j++) {
          R[i][j] = Number(parts[(i * 3 + j) + 10]);
        }
      }
      for (let i = 0; i < 3; i++) {
        R[i].push(Number(parts[i + 19]));
      }
    }
  });

  return matrix2ndMat((new Matrix(K)).mmul(new Matrix(R)));
};

const getNormalizeFunctions = (vertex: ndMat) => {

  let min_x = Number.MAX_SAFE_INTEGER;
  let max_x = Number.MIN_SAFE_INTEGER;

  let min_y = Number.MAX_SAFE_INTEGER;
  let max_y = Number.MIN_SAFE_INTEGER;

  let min_z = Number.MAX_SAFE_INTEGER;
  let max_z = Number.MIN_SAFE_INTEGER;

  for (let i = 3; i < vertex.length; i++) {
    min_x = Math.min(min_x, vertex[i][0]);
    min_y = Math.min(min_y, vertex[i][1]);
    min_z = Math.min(min_z, vertex[i][2]);

    max_x = Math.max(max_x, vertex[i][0]);
    max_y = Math.max(max_y, vertex[i][1]);
    max_z = Math.max(max_z, vertex[i][2]);
  }

  return {
    normalizeX: (x: number) => (x - min_x) / (max_x - min_x),
    normalizeY: (y: number) => (y - min_y) / (max_y - min_y),
    normalizeZ: (z: number) => (z - min_z) / (max_z - min_z)
  };
};

export const normalizeVertexList = (vertexList: ndMat) => {
  let normalizedVertexList: ndMat = [];
  const { normalizeX, normalizeY, normalizeZ } = getNormalizeFunctions(vertexList);
  vertexList.forEach(vertex => {
    normalizedVertexList.push([
      normalizeX(vertex[0]),
      normalizeY(vertex[1]),
      normalizeZ(vertex[2])
    ]);
  });
  return normalizedVertexList;
};