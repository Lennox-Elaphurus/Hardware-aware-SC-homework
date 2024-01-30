#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

const int P = 5;
const int cycles = 3;

int count = 0;
std::vector<int> flags(P,0);   // flags[i]==1 signals that thread i provides the result
std::vector<int> total_flags(P,0);
std::vector<std::mutex> ms(P); // mutexes
std::mutex mx;
std::vector<std::condition_variable> cvs(P); // and condition variables
std::mutex px; // mutex for protecting cout

void barrier (int rank)
{
  for (int stride=1; stride<P; stride*=2)
    if (rank%(2*stride)==0)
      {
        // add result from partner
        auto other = rank + stride;
        if (other<P)
          {
            std::unique_lock<std::mutex> lock{ms[other]};
            cvs[other].wait(lock,[other]{return flags[other]==1;});
            // 
            std::lock_guard<std::mutex> lg{px};
            std::cout << "Merge thread " << other << " into thread " << rank << "." << std::endl;
            flags[other] = 2; // reset flag
          }
      }
    else
      {
        // notify that result is ready
        std::unique_lock<std::mutex> lock{ms[rank]};
        flags[rank] = 1;
        cvs[rank].notify_one();
        break;
      }
  
  count += 1;
  std::unique_lock<std::mutex> total_lock{mx};
  if (count < P)
    {
      // wait on my cv until all have arrived
      total_flags[rank] = 1; // indicate I am waiting
      cvs[rank].wait(total_lock,[rank]{return total_flags[rank]==0;}); // wait
    }
  else
    {
      // I am the last one, lets wake them up
      count = 0; // reset counter for next turn
      std::cout << "------" << std::endl;
      for (int j = 0; j < P; j++)
	    if (total_flags[j]==1)
	    {
	        total_flags[j] = 0; // the event
	        cvs[j].notify_one(); // wake up
	    }
    }
}

void f (int i)
{
  for (int j=0; j<cycles; j++)
    {
      barrier(i); // block until all threads arrived
      std::lock_guard<std::mutex> lg{px};
      std::cout << "Thread " << i << std::endl;
    }
}

int main ()
{
  std::vector<std::thread> th;
  for (int i=0; i<P; ++i)
    th.push_back(std::thread{f,i});
  for (int i=0; i<P; ++i)
    th[i].join();
}