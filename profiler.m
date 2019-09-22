X = csvread('profiler.csv');
nRows = size(X, 1);
nSeek = 2;

# pkg install -forge outliers;
pkg load outliers;

tCapture = X(nSeek:nRows, 1);
tProc = X(nSeek:nRows, 2);
tVis = X(nSeek:nRows, 3);
tTotal = X(nSeek:nRows, 4);

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

showData(4, 1, tCapture, 'tCapture');
showData(4, 2, tProc, 'tProc');
showData(4, 3, tVis, 'tVis');
showData(4, 4, tTotal, 'tTotal');
