import random
import math
import pandas as pd
import matplotlib.pyplot as plt

# Memuat dataset dari CSV
file_path = 'D:\\data_jarak_antar kota_jawa timur.csv'
data = pd.read_csv(file_path, sep=";", on_bad_lines='skip')

# Mengubah dataset menjadi dictionary
def load_dataset():
    dataset = {}
    for index, row in data.iterrows():
        asal = row['Asal kota']
        tujuan = row['Kota tujuan']
        jarak = float(row['Jarak'].split()[0])  # Ekstrak nilai numerik dari 'Jarak'
        
        if asal not in dataset:
            dataset[asal] = {}
        dataset[asal][tujuan] = jarak
    
    return dataset

# Fungsi untuk mendapatkan informasi kota
def getCity():
    cities = [["Kota Surabaya", 0, 0]]  # Kota Surabaya sebagai titik pusat
    city_names = list(dataset["Kota Surabaya"].keys())
    for i, city_name in enumerate(city_names):
        distance = dataset["Kota Surabaya"][city_name]
        # Menggunakan koordinat polar untuk representasi visual
        angle = (i / len(city_names)) * 2 * math.pi
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        cities.append([city_name, x, y])
    return cities

# Fungsi untuk menghitung jarak antar kota
def calcDistance(cities):
    total_sum = 0
    for i in range(len(cities) - 1):
        cityA = cities[i]
        cityB = cities[i + 1]
        d = math.sqrt(math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))
        total_sum += d
    cityA = cities[0]
    cityB = cities[-1]
    d = math.sqrt(math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))
    total_sum += d
    return total_sum

# Fungsi untuk menghitung nilai fitness
def calcFitness(distance):
    return 1 / distance if distance != 0 else float('inf')

# Fungsi untuk memilih populasi
def selectPopulation(cities, size):
    population = []
    for i in range(size):
        c = cities.copy()
        random.shuffle(c[1:])
        # Pastikan Surabaya tetap sebagai titik awal dan akhir
        c = [cities[0]] + c[1:] + [cities[0]]
        distance = calcDistance(c)
        fitness = calcFitness(distance)
        population.append([distance, fitness, c])
    fittest = sorted(population)[0]
    return population, fittest

# Algoritma genetika
def geneticAlgorithm(population, lenCities, TOURNAMENT_SELECTION_SIZE, MUTATION_RATE, CROSSOVER_RATE, TARGET):
    gen_number = 0
    for i in range(200):
        new_population = []
        # Elitisme: Memilih dua individu terbaik langsung
        new_population.append(sorted(population)[0])
        new_population.append(sorted(population)[1])
        
        for i in range(int((len(population) - 2) / 2)):
            # Seleksi turnamen untuk memilih orang tua
            parent_chromosome1 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0]
            parent_chromosome2 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0]
            
            # Crossover
            random_number = random.random()
            if random_number < CROSSOVER_RATE:
                point = random.randint(1, lenCities - 2)  # Pastikan titik awal dan akhir tidak berubah
                child_chromosome1 = [parent_chromosome1[2][0]] + parent_chromosome1[2][1:point] + [j for j in parent_chromosome2[2] if j not in parent_chromosome1[2][1:point]] + [parent_chromosome1[2][0]]
                child_chromosome2 = [parent_chromosome2[2][0]] + parent_chromosome2[2][1:point] + [j for j in parent_chromosome1[2] if j not in parent_chromosome2[2][1:point]] + [parent_chromosome2[2][0]]
            # Jika crossover tidak terjadi
            else:
                child_chromosome1 = random.choices(population)[0][2]
                child_chromosome2 = random.choices(population)[0][2]
            
            # Mutasi
            if random.random() < MUTATION_RATE:
                point1 = random.randint(1, lenCities - 2)
                point2 = random.randint(1, lenCities - 2)
                child_chromosome1[point1], child_chromosome1[point2] = child_chromosome1[point2], child_chromosome1[point1]
                point1 = random.randint(1, lenCities - 2)
                point2 = random.randint(1, lenCities - 2)
                child_chromosome2[point1], child_chromosome2[point2] = child_chromosome2[point2], child_chromosome2[point1]
            
            # Penilaian fitness
            distance1 = calcDistance(child_chromosome1)
            fitness1 = calcFitness(distance1)
            distance2 = calcDistance(child_chromosome2)
            fitness2 = calcFitness(distance2)
            new_population.append([distance1, fitness1, child_chromosome1])
            new_population.append([distance2, fitness2, child_chromosome2])
        
        population = new_population
        gen_number += 1
        if gen_number % 10 == 0:
            print(f"Generation {gen_number}: Best Distance = {sorted(population)[0][0]}, Best Fitness = {sorted(population)[0][1]}")
        if sorted(population)[0][0] < TARGET:
            break
    
    answer = sorted(population)[0]
    return answer, gen_number

# Fungsi untuk menggambar peta kota dan hasil
def drawMap(city, answer):
    for j in city:
        plt.plot(j[1], j[2], "ro")
        plt.annotate(j[0], (j[1], j[2]))
    for i in range(len(answer[2]) - 1):
        first = answer[2][i]
        second = answer[2][i + 1]
        plt.plot([first[1], second[1]], [first[2], second[2]], "gray")
    first = answer[2][0]
    second = answer[2][-1]
    plt.plot([first[1], second[1]], [first[2], second[2]], "gray")
    plt.show()

def main():
    # Nilai awal
    POPULATION_SIZE = 2000
    TOURNAMENT_SELECTION_SIZE = 4
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    TARGET = 450.0
    cities = getCity()
    firstPopulation, firstFittest = selectPopulation(cities, POPULATION_SIZE)
    answer, genNumber = geneticAlgorithm(firstPopulation, len(cities), TOURNAMENT_SELECTION_SIZE, MUTATION_RATE, CROSSOVER_RATE, TARGET)
    print("\n----------------------------------------------------------------")
    print("Generation: " + str(genNumber))
    print("Fittest chromosome distance before training: " + str(firstFittest[0]))
    print("Fittest chromosome distance after training: " + str(answer[0]))
    print("Fitness before training: " + str(firstFittest[1]))
    print("Fitness after training: " + str(answer[1]))
    print("Target distance: " + str(TARGET))
    print("----------------------------------------------------------------\n")
    
    # Menampilkan rute terpendek
    route = " - ".join([city[0] for city in answer[2]])
    print("Rute terpendek: " + route)
    print("Total jarak terpendek: " + str(answer[0]) + " km")

    drawMap(cities, answer)

# Memuat dataset menjadi dictionary
dataset = load_dataset()
main()
