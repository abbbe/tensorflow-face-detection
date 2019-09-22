X = csvread('profiler.csv');
nRows = size(X, 1);
nSeek = 2;

# pkg install -forge outliers;
pkg load outliers;

tCapture = X(nSeek:nRows, 1);
tFaceDet = X(nSeek:nRows, 2);
tFaceProc = X(nSeek:nRows, 3);
tVis = X(nSeek:nRows, 4);
tTotal = X(nSeek:nRows, 5);

close all;
hold on;

function showData(rows, pos, data, name)
  m = mean(rmoutlier(data));
  mText = sprintf('avg=%.3f', m);
 
  subplot (rows, 2, pos*2 - 1);
  plot(data);
  legend(name);
  
  subplot (rows, 2, pos*2);
  hist(data);
  legend(mText);
endfunction

showData(5, 1, tCapture, 'tCapture');
showData(5, 2, tFaceDet, 'tFaceDet');
showData(5, 3, tFaceProc, 'tFaceProc');
showData(5, 4, tVis, 'tVis');
showData(5, 5, tTotal, 'tTotal');
