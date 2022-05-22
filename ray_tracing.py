import itertools
import numpy as np
import array

class Sphere:

    def __init__(self, color, ka, kd, ks, eta, kr, kt, n, center, radius) -> None:
        self.color = color
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.eta = eta
        self.kr = kr
        self.kt = kt
        self.n = n
        self.center = center
        self.radius = radius

    # encontra intersseção com a esfera
    def intersect(self, O, d):
        l = self.center - O
        t_ca = np.dot(l, d)
        d_squared = np.dot(l, l) - t_ca**2
        if d_squared > self.radius**2:
            raise Exception
        else:
            t_hc = (self.radius**2 - d_squared)**(1/2)
            t_0, t_1 = (t_ca - t_hc, t_ca + t_hc)
            if t_0 > t_1:
                t_0, t_1 = t_1, t_0
            if t_0 < 0:
                if t_1 < 0:
                    raise Exception
                else:
                    return t_1
            else:
                return t_0

    # retorna vetor unitário na direção refratada
    def refract(self, omega, normal):

        n = self.n
        cos_theta = np.dot(normal, omega)
        if cos_theta < 0:
            normal = -1*normal
            n = 1/n
            cos_theta = -cos_theta
        delta = 1 - (n**(-2))*(1 - cos_theta**2)
        if delta < 0:
            raise Exception
        return (-1/n)*omega - (delta**(1/2) -  cos_theta/n)*normal

    # retorna cor do ponto atingido
    def shade(self, P, omega, normal, background_light, lights, objects):

        cp = self.ka * self.color * background_light / 255
        for light_color, light_point in lights:
            l = (light_point - P)/np.linalg.norm(light_point - P)
            r = reflect(l, normal)
            _P = P + 0.00001*l
            S = trace(_P, l, objects)
            try:
                t, _ = min(S, key=lambda x: x[0])
            except:
                t = 0
            if (not S) or (np.dot(l, light_point - _P) < t):
                if np.dot(normal, l) > 0:
                    cp = cp + self.kd*self.color*np.dot(normal, l)*light_color / 255
                if np.dot(omega, r) > 0:
                    cp = cp + self.ks*(np.dot(omega, r)**self.eta)*light_color
        return cp

    # vetor unitário normal à superfície da esfera
    def normal_point(self, P):

        return (P - self.center)/np.linalg.norm(P - self.center)

class Plane:

    def __init__(self, color, ka, kd, ks, eta, kr, kt, n, point, normal) -> None:

        self.color = color
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.eta = eta
        self.kr = kr
        self.kt = kt
        self.n = n
        self.point = point
        self.normal = normal

    # encontra intersseção com o plano
    def intersect(self, O, d):

        den = np.dot(d, self.normal)
        if abs(den) > 10**-6:
            t = np.dot(self.point - O, self.normal)/den
            if t < 0:
                raise Exception
            else:
                return t
        else:
            raise Exception

    # retorna vetor unitário na direção refratada
    def refract(self, omega, normal):

        n = self.n
        cos_theta = np.dot(normal, omega)
        if cos_theta < 0:
            normal = -1*normal
            n = 1/n
            cos_theta = -cos_theta
        delta = 1 - (n**(-2))*(1 - cos_theta**2)
        if delta < 0:
            raise Exception
        return (-1/n)*omega - (delta**(1/2) -  cos_theta/n)*normal

    # retorna cor do ponto atingido
    def shade(self, P, omega, normal, background_light, lights, objects):

        cp = self.ka * self.color * background_light / 255
        for light_color, light_point in lights:
            l = (light_point - P)/np.linalg.norm(light_point - P)
            r = reflect(l, normal)
            _P = P + 0.00001*l
            S = trace(_P, l, objects)
            try:
                t, _ = min(S, key=lambda x: x[0])
            except:
                t = 0
            if (not S) or (np.dot(l, light_point - _P) < t):
                if np.dot(normal, l) > 0:
                    cp = cp + self.kd*self.color*np.dot(normal, l)*light_color / 255
                if np.dot(omega, r) > 0:
                    cp = cp + self.ks*(np.dot(omega, r)**self.eta)*light_color
        return cp

    # vetor unitário normal à superfície da esfera
    def normal_point(self, P):

        return self.normal/np.linalg.norm(self.normal)

# vetor unitário na direção refletida
def reflect(l, n):

    return 2*np.dot(n, l)*n - l

# retorna conjunto representando os objetos atingidos pelo raio
def trace(O, d, objects):

    S = []
    for obj in objects:
        try:
            t = obj.intersect(O, d)
            S.append([t, obj])
        except:
            pass
    return S

# age recursivamente -  retorna cor do objeto mais proximo atingido pelo raio
def cast(O, d, k, background_color, background_light, lights, objects):
    O = O + 0.00001*d
    cp = background_color
    S = trace(O, d, objects)
    if S:
        t, obj = min(S, key=lambda x: x[0])
        P = O + t*d
        omega = -1*d
        normal = obj.normal_point(P)
        cp = obj.shade(P, omega, normal, background_light, lights, objects)
        if k > 0:
            r = reflect(omega, normal)
            _P = P + 0.00001*r
            try:
                if obj.kt > 0:
                    r_t = obj.refract(omega, normal)
                    _P_t = P + 0.00001*r_t
                    cp = cp + obj.kt*cast(_P_t, r_t, k-1, background_color, background_light, lights, objects)
                if obj.kr > 0:
                    cp = cp + obj.kr*cast(_P, r, k-1, background_color, background_light, lights, objects)
            except:
                cp = cp + cast(_P, r, k-1, background_color, background_light, lights, objects)
    return np.array([min(255, elem) for elem in cp])

# inicia e junta a imagem através da chamada do cast
def render(v_res, h_res, s, d, e, l, up, max_depth,  background_color, background_light, lights, objects):
    img = np.array([[np.array([0, 0, 0])]*h_res]*v_res)

    w = e-l
    w = w/np.linalg.norm(w)

    u = np.cross(up, w)
    u = u/np.linalg.norm(u)

    v = np.cross(w, u)

    q0 = e-(d*w)+s*((v*((v_res-1)/2))-(u*((h_res-1)/2)))

    for (i, j) in itertools.product(range(v_res), range(h_res)):
        q = q0+s*((j*u)-(i*v))-e
        q = q/np.linalg.norm(q)
        img[i][j] = cast(e, q, max_depth, background_color, background_light, lights, objects)

    return img

if __name__ == '__main__':
    
    with open('input.txt', 'r') as f:

        # números de linhas e colunas
        v_res, h_res = [int(n) for n in f.readline().split(' ')]
        # tamanho do lado dos pixels e distancia focal
        s, d = [float(n) for n in f.readline().split(' ')]
        # foco
        e = np.array([float(n) for n in f.readline().split(' ')])
        # mira
        l = np.array([float(n) for n in f.readline().split(' ')])
        # vetor apontando pra cima
        up = np.array([float(n) for n in f.readline().split(' ')])
        # cor do plano de fundo
        background_color = np.array([float(n) for n in f.readline().split(' ')])
        # tamanho máximo da recursão
        max_depth = int(f.readline())
        # número de objetos
        k_objs = int(f.readline())
        objects = []

        for k in range(k_objs): # itera sobre os objetos para salvar seus atributos

            line = f.readline()
            split = line.split(' ')
            element = split[10]
            color = np.array([float(n) for n in split[:3]])
            ka = float(split[3])
            kd = float(split[4])
            ks = float(split[5])
            eta = float(split[6])
            # reflexão
            kr = float(split[7])
            # transparência e ́ındice de refração
            kt = float(split[8])
            n = float(split[9])

            # elemento é uma esfera 
            if element == '*':
                center = np.array([float(n) for n in split[11:14]])
                radius = float(split[14])
                sphere = Sphere(color, ka, kd, ks, eta, kr, kt, n, center, radius)
                objects.append(sphere)

            # elemento é um plano
            if element == '/':
                point = np.array([float(n) for n in split[11:14]])
                normal = np.array([float(n) for n in split[14:]])
                plane = Plane(color, ka, kd, ks, eta, kr, kt, n, point, normal)
                objects.append(plane)
        
        # luzes da cena
        background_light = np.array([float(n) for n in f.readline().split(' ')])
        # número de fontes pontuais
        k_lights = int(f.readline())
        lights = []

        for k in range(k_lights): # itera sobre as fontes pontuais de luz para salvar seus atributos
            
            line = f.readline()
            split = line.split(' ')
            # cor
            i = np.array([float(n) for n in split[:3]])
            # localização
            p = np.array([float(n) for n in split[3:]])
            _t = (i, p)
            lights.append(_t)
    
    # chama a funcao render pra retornar imagem (cor de cada pixel)
    img = render(v_res, h_res, s, d, e, l, up, max_depth,  background_color, background_light, lights, objects)
    with open('output.ppm', 'wb') as f:
        f.write(bytearray(f'P6 {h_res} {v_res} 255\n', 'ascii'))
        byteimg = array.array('B', list(img.flatten()))
        byteimg.tofile(f)
