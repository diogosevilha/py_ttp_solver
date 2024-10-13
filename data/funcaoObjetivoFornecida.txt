using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace TTPprogramm
{
    public class Item
    {
        public int Index;               // Item's index
        public long Profit;             // Item's profit
        public long Weight;             // Item's weight
        public bool Selected = false;   // False if the item is not picked in the packing plan, and true otherwise
    }

    public class City
    {
        public int Index;                           // Index of the city
        public List<Item> Items = new List<Item>(); // Set of items assigned to the city
        public int Distance;                        // Distance to the next city in the tour. REMARK: distance for Tour[n-1] corresponds to the distance between cities (n-1) and 0 of solution Tour
        public int m;                               // Number of items in the city
        public double PositionX;                    // Position X of the city
        public double PositionY;                    // Position Y of the city
    }

    public class Problem_TTP
    {
        static public City[] Cities;            // Set of cities in the instance
        static public int n;                    // Number of cities in the instance
        static public int mItems;               // Total number of items in the instance
        static public double R;                 // Renting rate
        static public double MinSpeed;          // Minimal speed
        static public double MaxSpeed;          // Maximal speed
        static public long MaxWeight;           // Maximal knapsack's weight

        // Function to compute a distance between two cities
        public static int GetDistance(City a, City b)
        {
            return (int)Math.Ceiling(Math.Sqrt(Math.Pow(a.PositionX - b.PositionX, 2) + Math.Pow(a.PositionY - b.PositionY, 2)));
        }

        // Function to Read TTP Instance
        static public bool ReadTTPInstance(string FileName)
        {
            try
            {
                using (StreamReader sr = File.OpenText(FileName))
                {
                    string input = sr.ReadLine();
                    while ((input != null) && (!input.Contains("NODE_COORD_SECTION")))
                    {
                        if (input.Contains("DIMENSION:")) int.TryParse(input.Replace("DIMENSION:", ""), out n);
                        if (input.Contains("NUMBER OF ITEMS")) int.TryParse(input.Replace("NUMBER OF ITEMS:", ""), out mItems);
                        if (input.Contains("RENTING RATIO:")) double.TryParse(input.Replace("RENTING RATIO:", ""), out R);
                        if (input.Contains("MIN SPEED:")) double.TryParse(input.Replace("MIN SPEED:", ""), out MinSpeed);
                        if (input.Contains("MAX SPEED:")) double.TryParse(input.Replace("MAX SPEED:", ""), out MaxSpeed);
                        if (input.Contains("CAPACITY OF KNAPSACK:")) long.TryParse(input.Replace("CAPACITY OF KNAPSACK:", ""), out MaxWeight);
                        input = sr.ReadLine();
                    }
                    input = sr.ReadLine(); // reading the NODE_COORD_SECTION part
                    Cities = new City[n];
                    for (int i = 0; i < n; i++)
                    {
                        Cities[i] = new City();
                        if (ReadCity(RemoveDoubleSpace(input.Trim().Replace("\t", " ")), ref Cities[i]))
                        {
                            input = sr.ReadLine();
                        }
                        else
                        {
                            throw new Exception("Error: inconsistent data in the ITEMS SECTION of the file");
                        }
                    }
                    input = sr.ReadLine(); // reading the NODE_COORD_SECTION part
                    for (int i = 0; i < mItems; i++)
                    {
                        Item item = new Item();
                        int NodeIndex = 0;
                        if (ReadItem(RemoveDoubleSpace(input.Trim().Replace("\t", " ")), ref item, ref NodeIndex)) 
                        {
                            Cities[NodeIndex - 1].Items.Add(item);
                            input = sr.ReadLine();
                        }
                        else
                        {
                            throw new Exception("Error: inconsistent data in the ITEMS SECTION of the file");
                        }
                    }
                }
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message);
                return false;
            }
            return true;
        }

        static private string RemoveDoubleSpace(string s)
        {
            string Result = s;
            while (Result.Contains("  ")) Result = Result.Replace("  ", " ");
            return Result;
        }

        static private bool ReadItem(string input, ref Item item, ref int NodeIndex)
        {
            try
            {
                string[] ptr;
                char[] b = new char[1] { ' ' };
                ptr = input.Split(b);
                if (ptr.Length < 3) return false;
                int.TryParse(ptr[0], out item.Index);
                long.TryParse(ptr[1], out item.Profit);
                long.TryParse(ptr[2], out item.Weight);
                int.TryParse(ptr[3], out NodeIndex);
                return true;
            }
            catch
            {
                return false;
            }
        }

        static private bool ReadCity(string input, ref City city)
        {
            try
            {
                string[] ptr;
                char[] b = new char[1] { ' ' };
                ptr = input.Split(b);
                if (ptr.Length != 3) return false;
                int.TryParse(ptr[0], out city.Index);
                double.TryParse(ptr[1], out city.PositionX);
                double.TryParse(ptr[2], out city.PositionY);
                return true;
            }
            catch
            {
                return false;
            }
        }
    }

    // Class represents solution through a sequence of cities Tour
    public class Solution
    {
        static public City[] Tour;  // Tour that represents the TTP solution. Tour[0] corresponds to the first city and has no items as stated in the TTP problem description.

        // Function to compute the objective value of the solution Tour
        static public double CalculateObjectiveValue(double MinSpeed, double MaxSpeed, long MaxWeight, double R)
        {
            long CollectedWeight = 0;   // Collected weight while traveling along Tour
            double ObjectiveValue = 0;
            double SpeedCoef = (MaxSpeed - MinSpeed) / MaxWeight;

            foreach (City city in Tour)
            {
                foreach (Item item in city.Items)
                {
                    if (item.Selected)
                    {
                        CollectedWeight += item.Weight;
                        ObjectiveValue += item.Profit;
                    }
                }
                ObjectiveValue -= city.Distance * R / (MaxSpeed - SpeedCoef * CollectedWeight);
            }
            return ObjectiveValue;
        }    
    }

    // Class represents an example. 
    public class Example
    {
        // Function constructs solution Tour by copying cities in their original sequence in Problem_TTP and sets the distancies between each pair of citites
        // Then it selects first item of the city 1 and computes the objective value.
        static public void TrialSolution()
        {
            Solution.Tour=Problem_TTP.Cities;
            for(int i=0; i<Problem_TTP.n-1; i++)
            {
                Solution.Tour[i].Distance = Problem_TTP.GetDistance(Solution.Tour[i], Solution.Tour[i + 1]);
            }
            Solution.Tour[Problem_TTP.n - 1].Distance = Problem_TTP.GetDistance(Solution.Tour[Problem_TTP.n - 1], Solution.Tour[0]);
            Solution.Tour[1].Items[0].Selected = true;

            Trace.WriteLine("Objective function: "+Solution.CalculateObjectiveValue(Problem_TTP.MinSpeed, Problem_TTP.MaxSpeed, Problem_TTP.MaxWeight, Problem_TTP.R));

        }
    }
}