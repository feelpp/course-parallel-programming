library(FactoMineR)
library(missMDA)
library(FactoInvestigate)

library(data.table)
library(FactoMineR)
library(caTools)
library(fpc)
library(numDeriv)
library(mclust)

library(mgcv)
library(nlme)
library(lme4)

library(e1071) 
library(moments)

library(compiler)
library(microbenchmark)

#require(compiler)
#enableJIT(3)


#install.packages("devtools")
library(devtools)
#install_github("bwrc/edf")
library(edf)
library(plotrix)

library(cluster)
library(vegan)

#install.packages("corrplot")
library(corrplot)
library(devtools)
#install_github("kassambara/factoextra")
library(PerformanceAnalytics)
library(factoextra)
library(Hmisc)

library(ggplot2)
library(plot3D)
library(rgl)

library(data.table)
library(FactoMineR)
library(caTools)
library(fpc)
library(numDeriv)
library(mclust)

#SOFTWARE
#library(Rcmdr)


library(cluster)
library(vegan)


#install.packages("corrplot")
library(corrplot)
library(devtools)
#install_github("kassambara/factoextra")
library(PerformanceAnalytics)
library(factoextra)
library(Hmisc)

library(ggplot2)
library(plot3D)
library(rgl)


library(leaps)

library(ggplot2)
#####

library(devtools)

library(heatmaply)
library(spatstat)
library("ape")


library("ggplot2")
library("ggdendro")

library(pca3d)

library("FactoMineR") 
library("factoextra")

library("plot3Drgl")
library(rgl)



Path<-"D:/ProgR/001/Data001/";

Path<-"D:/ProgR/001/Data005/";
#Path<-"D:/ProgR/001/Data005Gamma/";
Path<-"D:/ProgR/001/Data006/";
Flag<-"1A"
Flag<-"1B"
Flag<-"1"
Flag<-"10"
Flag<-"100"
Flag<-"1000"
Flag<-"10000"
Flag<-"500000"

Flag<-"100000A"
Flag<-"100000B"

Flag<-"100000"
Flag<-"200000"
Flag<-"300000"
Flag<-"400000"
Flag<-"500000"
Flag<-"500000B"

if (1==1) {
NameOutput=paste("Gaya","_",Flag,sep='');
NameInput<-paste("Mat",Flag,".csv",sep='');



FileLoad<-paste(Path,NameInput,sep='');
DATA<-read.table(FileLoad,sep=",",header=FALSE,dec=".");
LenFile=length(DATA[,1]);
#Lm=round(sqrt(LenFile))-1;
Lm=round(sqrt(LenFile));
MatDist=matrix(0.0,nrow = Lm, ncol = Lm);

for (k in (1:LenFile))
{
  i=DATA[k,1]+1;
  j=DATA[k,2]+1;
  Value=DATA[k,3];
  
  print(paste("i",i,"j",j," v=",Value,sep=''));
  
  MatDist[i,j]=Value;
}

if (0==1)
{
  MatDist3<-MatDist[-1,-1]
  MatDist<-MatDist3
  Lm=Lm-1
}

pca <- prcomp(MatDist, scale.= TRUE)
LabelName=1:Lm



old<-par();
#par(mfrow = c(1, 1))
HorizontalResolution<-2048;
VerticalResolution<-2048/2;
NameGraph<-paste(Path,NameOutput,"_Cluster_0.bmp",sep='');
bmp(NameGraph,
    width=trunc(HorizontalResolution), height=trunc(VerticalResolution), units = "px", pointsize = 12,
    bg = "white", res = NA, family = "", restoreConsole = TRUE,
    type = c("windows", "cairo"))

par(mfrow=c(2,2))

Md=MatDist
res.pca <- PCA(Md[,1:95], graph=FALSE)
hc <- HCPC(res.pca, nb.clust=-1)
plot(hc, choice="3D.map", angle=60)
plot(hc, choice="map")
#levels(hc$call$X$clust) <- c("Front1", "Front2", "Front3")
plot(hc, choice="map")

plot(hc, choice="3D.map", angle=60)
par(mfrow=c(1,1))
par(new=FALSE,ann=TRUE);
dev.off();


old<-par();
#par(mfrow = c(1, 1))
HorizontalResolution<-2048;
VerticalResolution<-2048/2;
NameGraph<-paste(Path,NameOutput,"_Cluster_1.bmp",sep='');
bmp(NameGraph,
    width=trunc(HorizontalResolution), height=trunc(VerticalResolution), units = "px", pointsize = 12,
    bg = "white", res = NA, family = "", restoreConsole = TRUE,
    type = c("windows", "cairo"))

par(mfrow=c(2,2))
Md=MatDist
res.pca <- PCA(Md[,1:95], graph=FALSE)
hc <- HCPC(res.pca, nb.clust=-1)
#plot(hc, choice="map")
#levels(hc$call$X$clust) <- c("Front1", "Front2", "Front3")
par(mfrow=c(1,1))
par(new=FALSE,ann=TRUE);
dev.off();




old<-par();
#par(mfrow = c(1, 1))
HorizontalResolution<-2048;
VerticalResolution<-2048/2;
NameGraph<-paste(Path,NameOutput,"_Cluster.bmp",sep='');
bmp(NameGraph,
    width=trunc(HorizontalResolution), height=trunc(VerticalResolution), units = "px", pointsize = 12,
    bg = "white", res = NA, family = "", restoreConsole = TRUE,
    type = c("windows", "cairo"))

par(mfrow=c(2,2))

hc = hclust(dist(MatDist))
plot(hc)
hcd <- as.dendrogram(hc)
#plot(hcd, type = "rectangle", ylab = "Height")

plot(as.phylo(hc), type = "unrooted", cex = 0.6,
     no.margin = TRUE)

#plot(as.phylo(hc), type = "fan")
colors = c("red", "blue", "green", "black")
clus4 = cutree(hc, 3)
plot(as.phylo(hc), type = "fan", tip.color = colors[clus4],
     label.offset = 0.1, cex = 0.7)
par(mfrow=c(1,1))
par(new=FALSE,ann=TRUE);
dev.off();



old<-par();
#par(mfrow = c(1, 1))
HorizontalResolution<-2048;
VerticalResolution<-2048/2;
NameGraph<-paste(Path,NameOutput,"_ClusterDendrogram.bmp",sep='');
bmp(NameGraph,
    width=trunc(HorizontalResolution), height=trunc(VerticalResolution), units = "px", pointsize = 12,
    bg = "white", res = NA, family = "", restoreConsole = TRUE,
    type = c("windows", "cairo"))

res.pca <- PCA(MatDist,ncp = 10, graph = FALSE)

res.hcpc <- HCPC(res.pca, graph = FALSE)

fviz_dend(res.hcpc, 
          cex = 0.7,                     # Label size
          palette = "jco",               # Color palette see ?ggpubr::ggpar
          rect = TRUE, rect_fill = TRUE, # Add rectangle around groups
          rect_border = "jco",           # Rectangle color
          labels_track_height = 0.8      # Augment the room for labels
)
par(mfrow=c(1,1))
par(new=FALSE,ann=TRUE);
dev.off();



old<-par();
#par(mfrow = c(1, 1))
HorizontalResolution<-2048;
VerticalResolution<-2048;
NameGraph<-paste(Path,NameOutput,"_ClusterDendrogram3D.bmp",sep='');
bmp(NameGraph,
    width=trunc(HorizontalResolution), height=trunc(VerticalResolution), units = "px", pointsize = 12,
    bg = "white", res = NA, family = "", restoreConsole = TRUE,
    type = c("windows", "cairo"))

plot(res.hcpc, choice = "3D.map",ind.names=TRUE,cex.axis=3.5,cex.symbols=3.5,xlab="",ylab="")
par(mfrow=c(1,1))
par(new=FALSE,ann=TRUE);
dev.off();


old<-par();
#par(mfrow = c(1, 1))
HorizontalResolution<-2048;
VerticalResolution<-2048/2;
NameGraph<-paste(Path,NameOutput,"_Plot.bmp",sep='');
bmp(NameGraph,
    width=trunc(HorizontalResolution), height=trunc(VerticalResolution), units = "px", pointsize = 12,
    bg = "white", res = NA, family = "", restoreConsole = TRUE,
    type = c("windows", "cairo"))

IMG <- im(MatDist)
plot(IMG)
par(mfrow=c(1,1))
par(new=FALSE,ann=TRUE);
dev.off();




old<-par();
#par(mfrow = c(1, 1))
HorizontalResolution<-2048;
VerticalResolution<-2048/2;
NameGraph<-paste(Path,NameOutput,"_Dendrogram.bmp",sep='');
bmp(NameGraph,
    width=trunc(HorizontalResolution), height=trunc(VerticalResolution), units = "px", pointsize = 12,
    bg = "white", res = NA, family = "", restoreConsole = TRUE,
    type = c("windows", "cairo"))

test=MatDist
par(mfrow=c(2,3))
plot(hclust(dist(test),method="average"))
plot(hclust(dist(test, method="maximum"),method="average"))
plot(hclust(dist(test, method="manhattan"),method="average"))
plot(hclust(dist(test, method="canberra"),method="average"))
plot(hclust(dist(test, method="minkowski", p=1.5),method="average"))
par(mfrow=c(1,1))
par(new=FALSE,ann=TRUE);
dev.off();


if (1==0)
{
  pca3d( pca, group= LabelName )
  pca3d(pca, group=LabelName,show.group.labels = TRUE)
}


library("plot3D")



hist3D_fancy<- function(x, y, break.func = c("Sturges", "scott", "FD"), breaks = NULL,
                        colvar = NULL, col="white", clab=NULL, phi = 5, theta = 25, ...){
  
  # Compute the number of classes for a histogram
  break.func <- break.func [1]
  if(is.null(breaks)){
    x.breaks <- switch(break.func,
                       Sturges = nclass.Sturges(x),
                       scott = nclass.scott(x),
                       FD = nclass.FD(x))
    y.breaks <- switch(break.func,
                       Sturges = nclass.Sturges(y),
                       scott = nclass.scott(y),
                       FD = nclass.FD(y))
  } else x.breaks <- y.breaks <- breaks
  
  # Cut x and y variables in bins for counting
  x.bin <- seq(min(x), max(x), length.out = x.breaks)
  y.bin <- seq(min(y), max(y), length.out = y.breaks)
  xy <- table(cut(x, x.bin), cut(y, y.bin))
  z <- xy
  
  xmid <- 0.5*(x.bin[-1] + x.bin[-length(x.bin)])
  ymid <- 0.5*(y.bin[-1] + y.bin[-length(y.bin)])
  
  oldmar <- par("mar")
  par (mar = par("mar") + c(0, 0, 0, 2))
  hist3D(x = xmid, y = ymid, z = xy, ...,
         zlim = c(-max(z)/2, max(z)), zlab = "counts", bty= "g", 
         phi = phi, theta = theta,
         shade = 0.2, col = col, border = "black",
         d = 1, ticktype = "detailed")
  
  scatter3D(x, y,
            z = rep(-max(z)/2, length.out = length(x)),
            colvar = colvar, col = gg.col(100),
            add = TRUE, pch = 18, clab = clab,
            colkey = list(length = 0.5, width = 0.5,
                          dist = 0.05, cex.axis = 0.8, cex.clab = 0.8)
  )
  par(mar = oldmar)
}


if (1==0)
{
MatDist2=MatDist

rownames(MatDist2)<-c(1:Lm)
colnames(MatDist2)<-c(1:Lm)

hist3D (x = 1:Lm, y = 1:Lm, z = MatDist2,
        bty = "g", phi = 20,  theta = -60,
        xlab = "", ylab = "", zlab = "", main = "Gaya",
        col = "cyan", border = "black", shade = 0.2,
        ticktype = "detailed", space = 0.15, d = 2, cex.axis = 1e-9)
plotrgl()
}




if (1==0)
{
  IMG <- im(MatDist)
  plot(IMG)
  
  Lm=round(sqrt(LenFile))-1;
  MatDist=matrix(0.0,nrow = Lm, ncol = Lm);
  for (k in (1:LenFile))
  {
    i=DATA[k,1];
    j=DATA[k,2];
    Value=DATA[k,3];
    MatDist[i,j]=Value;
  }
  
  
  scatter3D(DATA[,1],DATA[,2],DATA[,3], clab = c("Gaya", "Data"))
  
  scatter3D(DATA[,1],DATA[,2],DATA[,3], phi = 30, bty = "g",
            pch = 40, cex = 2, ticktype = "detailed")
  
  
  
  # Create a scatter plot
  scatter3D(DATA[,1],DATA[,2],DATA[,3], phi = 0, bty = "g", pch = 20, cex = 0.5)
  # Add text
  text3D(DATA[,1],DATA[,2],DATA[,3],  labels = rownames(1:LenFile),add = TRUE, colkey = FALSE, cex = 0.5)
  
  
  
  
  # type ="h" for vertical lines
  scatter3D(DATA[,1],DATA[,2],DATA[,3], phi = 0, bty = "g",  type = "h", 
            ticktype = "detailed", pch = 19, cex = 0.5)
  
  plotrgl()
  

  
  
  
  
  MatDist2=MatDist
  
  rownames(MatDist2)<-c(1:Lm)
  colnames(MatDist2)<-c(1:Lm)
  
  hist3D (x = 1:Lm, y = 1:Lm, z = MatDist2,
          bty = "g", phi = 20,  theta = -60,
          xlab = "", ylab = "", zlab = "", main = "Gaya",
          col = "cyan", border = "black", shade = 0.2,
          ticktype = "detailed", space = 0.15, d = 2, cex.axis = 1e-9)
  plotrgl()
  
  
  
  
  barplot(MatDist[,1])
  plot(1:Lm,MatDist[,1])
  plot(1:Lm,MatDist[,2])
  
  
  
  boxplot(MatDist[,])
  
  which(MatDist==max(MatDist), arr.ind=T)
  
  
  # Create his3D using plot3D
  hist3D_fancy(1:Lm,1:Lm, colvar=as.numeric(MatDist2))
  
  hist3D_fancy(Lm,Lm, colvar=as.numeric(MatDist2))
  #hist3D_fancy(iris$Sepal.Length, iris$Petal.Width, colvar=as.numeric(iris$Species))
  plotrgl()

}



boxplot(MatDist[,])

M=MatDist;

max =which(M==max(M), arr.ind=T)
print(paste("Maximum value: ",M[max],"in position row=",max[1,1],"col=",max[1,2]))
#xy <- as.vector(which(MatDist==max(M), arr.ind=T))
L0=c(M[max])
for (k in (1:200))
{
  M[max]=-1;
  max =which(M==max(M), arr.ind=T)
  print(paste("Maximum value: ",M[max],"in position row=",max[1,1],"col=",max[1,2]))
  L0=c(L0,M[max])
}
plot(L0)
}

