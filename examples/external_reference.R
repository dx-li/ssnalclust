# Run the separately installed, unmodified cvxclustr reference implementation.
# This bridge is original project code; it does not vendor cvxclustr sources.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 7)
input <- args[1]
output <- args[2]
gamma <- as.double(args[3])
type <- as.integer(args[4])
accelerate <- as.logical(args[5])
tol <- as.double(args[6])
max_iter <- as.integer(args[7])
suppressPackageStartupMessages(library(cvxclustr))
X <- t(as.matrix(read.csv(file.path(input, "X.csv"), header=FALSE)))
W <- as.matrix(read.csv(file.path(input, "W.csv"), header=FALSE))
n <- ncol(X)
# Upper triangle in lexicographic (i,j) order, including zero weights.
w <- unlist(lapply(seq_len(n-1), function(i) W[i, seq.int(i+1,n)]))
edge <- cvxclustr:::compactify_edges(w,n,method="ama")
Lambda <- matrix(0,nrow(X),sum(w>0))
started <- proc.time()["elapsed"]
fit <- cvxclust_ama(X,Lambda,edge$ix-1,edge$M1-1,edge$M2-1,
                   edge$s1,edge$s2,w[w>0],gamma,nu=1/n,type=type,
                   max_iter=max_iter,tol=tol,accelerate=accelerate)
elapsed <- proc.time()["elapsed"]-started
write_matrix <- function(value, filename) {
  encoded <- matrix(sprintf("%.17g", value), nrow=nrow(value))
  write.table(encoded,file.path(output,filename),sep=",",row.names=FALSE,col.names=FALSE,quote=FALSE)
}
write_matrix(t(fit$U),"centers.csv")
write_matrix(t(fit$Lambda),"lambda.csv")
write.table(edge$ix,file.path(output,"edges.csv"),sep=",",row.names=FALSE,col.names=FALSE)
metadata <- c(version=as.character(packageVersion("cvxclustr")),
              R=R.version.string,platform=R.version$platform,
              dll_md5=unname(tools::md5sum(getLoadedDLLs()[["cvxclustr"]][["path"]])),
              seconds=as.character(elapsed),iterations=as.character(fit$iter),
              native_primal=as.character(tail(fit$primal,1)),
              native_dual=as.character(tail(fit$dual,1)))
write.table(data.frame(name=names(metadata),value=unname(metadata)),
            file.path(output,"metadata.csv"),sep=",",row.names=FALSE)
