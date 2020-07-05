import { Mat } from 'opencv4nodejs';

export type IPoint = Array<number>;

export interface IMatchPoint {
  p1: IPoint;
  p2: IPoint;
}

type cvMat = Mat;

type ndMat = Array<Array<number>>;