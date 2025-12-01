#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <memory>
#include <mpi.h>
#include <vector>
#include <bit>
#include <chrono>
#define BUFSIZE 1024

void add_int(void*a,void*b,void*c){
  *(int*)c=*(int*)a+*(int*)b;
}

void(*dispatch_pred(MPI_Datatype datatype, MPI_Op ope))(void*,void*,void*){
  if(datatype==MPI_INT && ope==MPI_SUM) return add_int;
  return NULL;
}

auto AllReduceInfo(MPI_Comm comm, MPI_Datatype datatype, MPI_Op ope){
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  int type_size;
  if(MPI_SUCCESS!=MPI_Type_size(datatype, &type_size))
    type_size=0;
  void(*pred)(void*,void*,void*)=dispatch_pred(datatype,ope);
  return std::make_tuple(rank,size,type_size,pred);
}

int AllReduce_Linear(void* sendbuf, void*recvbuf, int count, MPI_Datatype datatype, MPI_Op ope, MPI_Comm comm){
  static int tag=0;
  auto [rank, size, type_size, pred]=AllReduceInfo(comm,datatype,ope);
  if(!type_size||!pred) return MPI_ERR_ARG;
  const size_t bufsize=type_size*count;
  if(rank==0){
    if(sendbuf!=MPI_IN_PLACE) // recvbufを計算の途中結果に利用
      memcpy(recvbuf, sendbuf, bufsize);
    std::vector<MPI_Request> req(size-1);
    std::vector recvbuf_peer(size-1,std::vector<char>(bufsize));
    for(int i=1;i<size;++i)
      MPI_Irecv(recvbuf_peer[i-1].data(), count, datatype, i, tag, comm, &req[i-1]);
    MPI_Waitall(size-1, req.data(), MPI_STATUS_IGNORE);
    for(int i=1;i<size;++i)
      for(int j=0;j<count;++j)
        pred(recvbuf_peer[i-1].data()+j*type_size, (char*)recvbuf+j*type_size, (char*)recvbuf+j*type_size);
    for(int i=1;i<size;++i)
      MPI_Isend(recvbuf, count, datatype, i, tag, comm, &req[i-1]);
    MPI_Waitall(size-1, req.data(), MPI_STATUS_IGNORE);
  }else{
    MPI_Request req;
    MPI_Isend(MPI_IN_PLACE==sendbuf?recvbuf:sendbuf, count, datatype, 0, tag, comm, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    MPI_Irecv(recvbuf, count, datatype, 0, tag, comm, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
  }
  ++tag;
  return MPI_SUCCESS;
}

int AllReduce_Square(void* sendbuf, void*recvbuf, int count, MPI_Datatype datatype, MPI_Op ope, MPI_Comm comm){
	static int tag=0;
  auto [rank, size, type_size, pred]=AllReduceInfo(comm,datatype,ope);
  if(!type_size||!pred) return MPI_ERR_ARG;

	char* sendbuf_local=reinterpret_cast<char*>(sendbuf==MPI_IN_PLACE?recvbuf:sendbuf);
	char* reduced_buf=reinterpret_cast<char*>(sendbuf==MPI_IN_PLACE?malloc(type_size*count):recvbuf);
	memset(reduced_buf, 0, count*type_size);
  char** recvbuf_local=(char**)malloc(sizeof(char*)*size);
	for(int i=0;i<size;++i)
		recvbuf_local[i]=(char*)malloc(type_size*count);

  MPI_Request *req=(MPI_Request*)malloc(2*size*sizeof(MPI_Request));

  for(int i=0;i<size;++i){
    MPI_Isend(sendbuf_local, count, datatype, i, tag, comm, req+i);
    MPI_Irecv(recvbuf_local[i], count, datatype, i, tag, comm, req+size+i);
	}
  MPI_Waitall(2*size, req, MPI_STATUS_IGNORE);
  for(int i=0;i<size;++i)
    for(int j=0;j<count;++j)
      pred(recvbuf_local[i]+j*type_size, reduced_buf+j*type_size, reduced_buf+j*type_size);
  if(sendbuf==MPI_IN_PLACE){
		memcpy(recvbuf, reduced_buf, count*type_size);
		free(reduced_buf);
	}
	{ // destructor
		++tag;
		free(req);
		for(int i=0;i<size;++i)
			free(recvbuf_local[i]);
		free(recvbuf_local);
	}
  return MPI_SUCCESS;
}

int AllReduce_Ring(void* sendbuf, void*recvbuf, int count, MPI_Datatype datatype, MPI_Op ope, MPI_Comm comm){
	static int tag=0;
  auto [rank, size, type_size, pred]=AllReduceInfo(comm,datatype,ope);
  if(!type_size||!pred) return MPI_ERR_ARG;
  const int bufsize=count*type_size;
  if(MPI_IN_PLACE!=sendbuf)
    memcpy(recvbuf, sendbuf, bufsize);
  const int prev = (rank+size-1)%size;
  const int next = (rank+1)%size;
  std::unique_ptr<char[]> to_send(new char[bufsize]);
  std::unique_ptr<char[]> to_recv(new char[bufsize]);
  MPI_Request req[2];
  memcpy(to_send.get(), recvbuf, bufsize);
  for(int i=0;i<size-1;++i){
    MPI_Isend(to_send.get(), count, datatype, next, tag, comm, req+0);
    MPI_Irecv(to_recv.get(), count, datatype, prev, tag, comm, req+1);
    MPI_Waitall(2, req, MPI_STATUS_IGNORE);
    for(int j=0;j<count;++j)
      pred(to_recv.get()+j*type_size, (char*)recvbuf+j*type_size, (char*)recvbuf+j*type_size);
    std::swap(to_send, to_recv);
  }
  return MPI_SUCCESS;
}

int AllReduce_Tree(void* sendbuf, void*recvbuf, int count, MPI_Datatype datatype, MPI_Op ope, MPI_Comm comm){
	static int tag=0;
  auto [rank, size, type_size, pred]=AllReduceInfo(comm,datatype,ope);
  if(!type_size||!pred) return MPI_ERR_ARG;
  const int parent = (rank-1>>1);
  const int child = (rank<<1)+1;
  MPI_Request req[2];
  char* to_recv=(char*)(MPI_IN_PLACE==sendbuf?malloc(type_size*count):recvbuf);
  char* to_send=(char*)(MPI_IN_PLACE==sendbuf?recvbuf:sendbuf);
  // 子から受け取る
  for(int i=0;i<2;++i){
    if(child+i>=size) break;
    MPI_Irecv(to_recv, count, datatype, child+i, tag, comm, req+i);
    MPI_Wait(req+i, MPI_STATUS_IGNORE);
    for(int j=0;j<count;++j)
      pred(to_recv+j*type_size, to_send+j*type_size, to_send+j*type_size);
  }
  // 根ならここで確定
  if(rank!=0){
  // 親に送る
    MPI_Isend(to_send, count, datatype, parent, tag, comm, req+0);
    MPI_Wait(req+0, MPI_STATUS_IGNORE);
  // 親から受け取る->これが答え
    MPI_Irecv(to_recv, count, datatype, parent, tag+1, comm, req+0);
    MPI_Wait(req+0, MPI_STATUS_IGNORE);
  // 根以外のときここで確定
  }
  // 答えをrecvbufに入れる
  // MPI_IN_PLACEのとき，recvbufに答えが入っているため何もしなくていい
  // 根のとき子から受け取った物が答え，そうでないとき，親から受け取った物が答え
  if(MPI_IN_PLACE!=sendbuf)
    memcpy(recvbuf, (rank==0?to_send:to_recv), count*type_size);
  // 子に送る
  for(int i=0;i<2;++i){
    if(child+i>=size) break;
    MPI_Isend(recvbuf, count, datatype, child+i, tag+1, comm, req+i);
  }
  MPI_Waitall((child<size)+(child+1<size), req, MPI_STATUS_IGNORE);
  if(MPI_IN_PLACE==sendbuf)
    free(to_recv);
  tag+=2;
  return MPI_SUCCESS;
}

int AllReduce_HyperCube(void* sendbuf, void*recvbuf, int count, MPI_Datatype datatype, MPI_Op ope, MPI_Comm comm){
	static int tag=0;
  auto [rank, size, type_size, pred]=AllReduceInfo(comm,datatype,ope);
  if(!type_size||!pred) return MPI_ERR_ARG;

  const int nofit=std::popcount((unsigned)size)!=1;
  const unsigned fullsize = (std::bit_ceil((unsigned int)size)>>nofit);
  const unsigned dim = std::popcount(fullsize-1);
  const int bufsize = count * type_size;

  if(MPI_IN_PLACE != sendbuf)
    memcpy(recvbuf, sendbuf, bufsize);
  
  if(size<=1) return MPI_SUCCESS;
  
  char* commbuf=(char*)malloc(bufsize);
  MPI_Request req[2];

  if(nofit){
    if(fullsize<=rank){ // ハイパーキューブ外
      MPI_Isend(recvbuf, count, datatype, rank-fullsize, tag, comm, req+0);
      MPI_Wait(req+0, MPI_STATUS_IGNORE);
    }else if(rank+fullsize<size){ // ハイパーキューブ内
      MPI_Irecv(commbuf, count, datatype, rank+fullsize, tag, comm, req+0);
      MPI_Wait(req+0, MPI_STATUS_IGNORE);
      for(int j=0;j<count;++j)
        pred(commbuf+j*type_size, (char*)recvbuf+j*type_size, (char*)recvbuf+j*type_size);
    }
  }
  
  if(rank<fullsize)
  for(int i=0;i<dim;++i){
    const int partner = rank ^ (1<<i);
    MPI_Isend(recvbuf, count, datatype, partner, tag, comm, req+0);
    MPI_Irecv(commbuf, count, datatype, partner, tag, comm, req+1);
    MPI_Waitall(2, req, MPI_STATUS_IGNORE);
    for(int j=0;j<count;++j)
      pred(commbuf+j*type_size, (char*)recvbuf+j*type_size, (char*)recvbuf+j*type_size);
  }
  
  if(nofit){
    if(fullsize<=rank)
      MPI_Irecv(recvbuf, count, datatype, rank-fullsize, tag+1, comm, req+0);
    else if(rank+fullsize<size)
      MPI_Isend(recvbuf, count, datatype, rank+fullsize, tag+1, comm, req+0);
    MPI_Wait(req+0, MPI_STATUS_IGNORE);
  }

  tag+=2;
  return MPI_SUCCESS;
}

int (*which(const char* name))(void*,void*,int,MPI_Datatype,MPI_Op,MPI_Comm){
  if(std::string_view("square")==name) return AllReduce_Square;
  else if(std::string_view("linear")==name) return AllReduce_Linear;
  else if(std::string_view("ring")==name) return AllReduce_Ring;
  else if(std::string_view("tree")==name) return AllReduce_Tree;
  else if(std::string_view("hypercube")==name) return AllReduce_HyperCube;
  return NULL;
}

int main(int argc, char**argv){
  MPI_Init(&argc, &argv);
  int* sendbuf=(int*)malloc(sizeof(int)*BUFSIZE);
  int* recvbuf=(int*)malloc(sizeof(int)*BUFSIZE);
  int* recvbuf_ans=(int*)malloc(sizeof(int)*BUFSIZE);
  for(int i=0;i<BUFSIZE;++i) sendbuf[i]=rand()%10;
  MPI_Allreduce(sendbuf, recvbuf_ans, BUFSIZE, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  auto f=argc>1?which(argv[1]):AllReduce_HyperCube;
  // rank 0 に実行モードを表示させる
  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(rank==0){
    if(argc==2) printf("execmode: %s\n", argv[1]);
    else printf("execmode: hypercube\n");
    printf("size: %d\n", size);
  }

  auto start=std::chrono::high_resolution_clock::now();
  f(sendbuf, recvbuf, BUFSIZE, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  auto end=std::chrono::high_resolution_clock::now();
  auto duration=std::chrono::duration_cast<std::chrono::microseconds>(end-start);
  printf("Time: %ld us\n", duration.count());

  MPI_Finalize();
  int is_different=0;
  for(int i=0;i<BUFSIZE;++i) is_different|=recvbuf[i]^recvbuf_ans[i];
  printf("%s%d ",is_different?"NG":"OK", rank);

  { // destructor
    free(sendbuf);
    free(recvbuf);
    free(recvbuf_ans);
  }
  return is_different;
}