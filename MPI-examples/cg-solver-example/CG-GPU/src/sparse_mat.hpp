#ifndef MPI_SPARSE_MAT_HPP
#define MPI_SPARSE_MAT_HPP

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <rccl/rccl.h>

#include <vector>

struct Mat
{
    std::vector<int> rowptr;
    std::vector<int> col_idx;
    std::vector<double> data;
    int n_rows;
    int n_cols;
    int nnz;
};

struct Comm
{
    int n_msgs;
    int size_msgs;
    std::vector<int> procs;
    std::vector<int> ptr;
    std::vector<int> counts;
    std::vector<int> idx;
    std::vector<MPI_Request> req;
};

struct ParMat
{
    Mat on_proc;
    Mat off_proc;
    int global_rows;
    int global_cols;
    int local_rows;
    int local_cols;
    int first_row;
    int first_col;
    int off_proc_num_cols;
    std::vector<long> off_proc_columns;
    Comm send_comm;
    Comm recv_comm;
    MPI_Comm dist_graph_comm;
};

void form_recv_comm(ParMat& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Gather first col for all processes into list
    std::vector<int> first_cols(num_procs + 1);
    MPI_Allgather(
        &A.first_col, 1, MPI_INT, first_cols.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_cols[num_procs] = A.global_cols;

    // Map Columns to Processes
    int proc      = 0;
    int prev_proc = -1;
    for (int i = 0; i < A.off_proc_num_cols; i++)
    {
        int global_col = A.off_proc_columns[i];
        while (first_cols[proc + 1] <= global_col)
        {
            proc++;
        }
        if (proc != prev_proc)
        {
            A.recv_comm.procs.push_back(proc);
            A.recv_comm.ptr.push_back((i));
            prev_proc = proc;
        }
    }

    // Set Recv Sizes
    A.recv_comm.ptr.push_back((A.off_proc_num_cols));
    A.recv_comm.n_msgs    = A.recv_comm.procs.size();
    A.recv_comm.size_msgs = A.off_proc_num_cols;
    if (A.recv_comm.n_msgs == 0)
    {
        return;
    }

    A.recv_comm.req.resize(A.recv_comm.n_msgs);
    A.recv_comm.counts.resize(A.recv_comm.n_msgs);
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        A.recv_comm.counts[i] = A.recv_comm.ptr[i + 1] - A.recv_comm.ptr[i];
    }
}

// Must Form Recv Comm before Send!
void form_send_comm_standard(ParMat& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<long> recv_buf;
    std::vector<int> sizes(num_procs, 0);
    int proc, count, ctr;
    MPI_Status recv_status;

    // Allreduce to find size of data I will receive
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        sizes[A.recv_comm.procs[i]] = A.recv_comm.ptr[i + 1] - A.recv_comm.ptr[i];
    }
    MPI_Allreduce(
        MPI_IN_PLACE, sizes.data(), num_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    A.send_comm.size_msgs = sizes[rank];

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int msg_tag = 1234;
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        MPI_Isend(&(A.off_proc_columns[A.recv_comm.ptr[i]]),
                  A.recv_comm.counts[i],
                  MPI_LONG,
                  proc,
                  msg_tag,
                  MPI_COMM_WORLD,
                  &(A.recv_comm.req[i]));
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    if (A.send_comm.size_msgs)
    {
        A.send_comm.idx.resize(A.send_comm.size_msgs);
        recv_buf.resize(A.send_comm.size_msgs);
    }
    ctr = 0;
    A.send_comm.ptr.push_back(0);
    while (ctr < A.send_comm.size_msgs)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, msg_tag, MPI_COMM_WORLD, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        A.send_comm.procs.push_back(proc);
        MPI_Get_count(&recv_status, MPI_LONG, &count);
        A.send_comm.counts.push_back(count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(recv_buf[ctr]),
                 count,
                 MPI_LONG,
                 proc,
                 msg_tag,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        for (int i = 0; i < count; i++)
        {
            A.send_comm.idx[ctr + i] = (recv_buf[ctr + i] - A.first_col);
        }
        ctr += count;
        A.send_comm.ptr.push_back((ctr));
    }

    // Set send sizes
    A.send_comm.n_msgs = A.send_comm.procs.size();

    if (A.send_comm.n_msgs)
    {
        A.send_comm.req.resize(A.send_comm.n_msgs);
    }

    if (A.recv_comm.n_msgs)
    {
        MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
    }
}

void form_comm(ParMat& A)
{
    // Form Recv Side
    form_recv_comm(A);

    // Form Send Side (Algorithm Options Here!)
    form_send_comm_standard(A);
}

// =============================================================================
// GPUMat — one sparse block (on-proc or off-proc) stored on the GPU
//
// Holds:
//   - Device-side CSR arrays  (d_rowptr, d_colidx, d_data)
//   - A rocSPARSE generic matrix descriptor
//   - A workspace buffer that rocSPARSE SpMV needs internally
// =============================================================================
struct GPUMat {
    int     n_rows, n_cols, nnz;
    int    *d_rowptr, *d_colidx;
    double *d_data;
    rocsparse_spmat_descr descr;      // rocSPARSE generic CSR descriptor
    void   *d_spmv_buf;               // rocSPARSE workspace (queried once)
    size_t  spmv_buf_size;

    // Persistent dense-vector descriptors — created once at upload and reused on
    // every SpMV (the value pointer is swapped in with rocsparse_dnvec_set_values).
    // Avoids create/destroy of two descriptors on every iteration's hot path.
    rocsparse_dnvec_descr vec_x, vec_b;

#ifdef USE_V2_SPMV
    // rocsparse_v2_spmv state (ROCm 7.x): a persistent SpMV descriptor whose
    // symbolic analysis stage is run once at upload and amortized over the solve.
    rocsparse_spmv_descr  spmv_v2;
    void                 *d_spmv_buf_a;   // analysis-stage workspace
#endif
};

// =============================================================================
// GPUParMat — the full distributed matrix on the GPU
//
// Same logical structure as ParMat above, but every numerical array lives on
// the device.  Holds all state needed by the four parallel SpMV variants so
// each variant has a uniform call signature.
// =============================================================================
struct GPUParMat {
    GPUMat  on_proc;       // diagonal block (columns owned by this rank)
    GPUMat  off_proc;      // off-diagonal block (ghost columns)

    // ── Variant 1 & 3: GPU buffers (GPU-Aware MPI / RCCL) ─────────────────
    double *d_sendbuf;     // packed values to send neighbours (GPU memory)
    double *d_recvbuf;     // ghost values received from neighbours (GPU memory)
    int    *d_send_idx;    // send index list on GPU (used by rocsparse_dgthr)

    // ── Variant 2: host buffers (staged CPU comm) ──────────────────────────
    // Pinned (page-locked) host memory for fast PCIe DMA transfers.
    double *h_sendbuf;
    double *h_recvbuf;

    // ── Variant 6: plain-malloc host buffers (staged_unified, MI300A APU) ───
    // Ordinary system memory (malloc). On an APU with HSA_XNACK=1 the GPU can
    // read/write these via page faults, while MPI sees them as HOST pointers and
    // uses the host transport -- zero copies AND no GPU-Aware MPI.
    double *u_sendbuf;
    double *u_recvbuf;

    // ── Variant 3: MPI_Alltoallv arrays (CPU, one entry per rank) ──────────
    // Non-communicating ranks simply have count = 0.
    std::vector<int> a2a_sendcounts, a2a_sdispls;
    std::vector<int> a2a_recvcounts, a2a_rdispls;

    // ── Variant 4: RCCL communicator and dedicated stream ──────────────────
    // RCCL runs on rccl_stream independently of the default ROCm stream,
    // enabling GPU-GPU overlap between communication and on-proc SpMV.
    ncclComm_t  rccl_comm;
    hipStream_t rccl_stream;

    // CPU comm metadata (procs[], ptr[], req[]) — used for MPI metadata
    Comm   *send_comm;
    Comm   *recv_comm;
};

#endif
