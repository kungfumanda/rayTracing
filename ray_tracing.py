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
        
        result = (-1/n)*omega - (delta**(1/2) -  cos_theta/n)*normal
        return result

    # retorna cor do ponto atingido
    def shade(self, P, omega, normal, background_light, pls, objects):

        color_point = self.ka * self.color * background_light / 255
        for light_color, light_point in pls:
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
                    color_point = color_point + self.kd*self.color*np.dot(normal, l)*light_color / 255
                if np.dot(omega, r) > 0:
                    color_point = color_point + self.ks*(np.dot(omega, r)**self.eta)*light_color
        return color_point
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

        result = (-1/n)*omega - (delta**(1/2) -  cos_theta/n)*normal
        return result

    # retorna cor do ponto atingido
    def shade(self, P, omega, normal, background_light, pls, objects):

        color_point = self.ka * self.color * background_light / 255
        for light_color, light_point in pls:
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
                    color_point = color_point + self.kd*self.color*np.dot(normal, l)*light_color / 255
                if np.dot(omega, r) > 0:
                    color_point = color_point + self.ks*(np.dot(omega, r)**self.eta)*light_color
        return color_point

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
def cast(O, d, k, background_color, background_light, pls, objects):
    color_point = background_color
    S = trace(O, d, objects)
    if S:
        t, obj = min(S, key=lambda x: x[0])
        P = O + t*d
        omega = -1*d
        normal = obj.normal_point(P)
        color_point = obj.shade(P, omega, normal, background_light, pls, objects)
        if k > 0:
            r = reflect(omega, normal)
            _P = P + 0.00001*r
            try:
                if obj.kt > 0:
                    r_t = obj.refract(omega, normal)
                    _P_t = P + 0.00001*r_t
                    color_point = color_point + obj.kt*cast((_P_t  + 0.00001*r_t), r_t, k-1, background_color, background_light, pls, objects)
                if obj.kr > 0:
                    color_point = color_point + obj.kr*cast((_P + 0.00001*r), r, k-1, background_color, background_light, pls, objects)
            except:
                color_point = color_point + cast((_P + 0.00001*r), r, k-1, background_color, background_light, pls, objects)
    return np.array([min(255, elem) for elem in color_point])


# inicia e junta a imagem através da chamada do cast
def render(v_res, h_res, s, d, e, l, up, max_depth,  background_color, background_light, pls, objects):
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
        img[i][j] = cast((e + 0.00001*q), q, max_depth, background_color, background_light, pls, objects)

    return img


# main
if __name__ == '__main__':

    with open('input.txt', 'r') as file:

        ''' parâmetros da câmera '''
        # números de linhas e colunas
        v_res, h_res = [int(n) for n in file.readline().split(' ')]
        # tamanho do lado dos pixels e distancia focal
        s, d = [float(n) for n in file.readline().split(' ')]
        # foco
        e = np.array([float(n) for n in file.readline().split(' ')])
        # mira
        l = np.array([float(n) for n in file.readline().split(' ')])
        # vetor apontando pra cima
        up = np.array([float(n) for n in file.readline().split(' ')])
        # cor do plano de fundo
        background_color = np.array([float(n) for n in file.readline().split(' ')])
        # tamanho máximo da recursão
        max_depth = int(file.readline())
        # número de objetos
        k_obj = int(file.readline())
        
        objects = []


        for _ in range(k_obj): # itera sobre os objetos para salvar seus atributos

            attributes = file.readline().split(' ')
            color = np.array([float(x) for x in attributes[:3]])
            ka = float(attributes[3])
            kd = float(attributes[4])
            ks = float(attributes[5])
            eta = float(attributes[6])
            # reflexão
            kr = float(attributes[7])
            # transparência e ́ındice de refração
            kt = float(attributes[8])
            n = float(attributes[9])

            element = attributes[10]
            
            # elemento é uma esfera 
            if element == '*':

                center = np.array([float(n) for n in attributes[11:14]])
                radius = float(attributes[14])
                sphere = Sphere(color, ka, kd, ks, eta, kr, kt, n, center, radius)
                objects.append(sphere)
            
            # elemento é um plano
            if element == '/':

                point = np.array([float(n) for n in attributes[11:14]])
                normal = np.array([float(n) for n in attributes[14:]])
                plane = Plane(color, ka, kd, ks, eta, kr, kt, n, point, normal)
                objects.append(plane)
        
        # luzes da cena
        background_light = np.array([float(n) for n in file.readline().split(' ')])
        # número de fontes pontuais
        k_pl = int(file.readline())

        pls = []

        for _ in range(k_pl): # itera sobre as fontes pontuais de luz para salvar seus atributos
            
            attributes = file.readline().split(' ')
            # cor
            c = np.array([float(n) for n in attributes[:3]])
            # localização
            l = np.array([float(n) for n in attributes[3:]])
            pl = (c, l)
            pls.append(pl)
    
    # chama a funcao render pra retornar imagem (cor de cada pixel)
    pic = render(v_res, h_res, s, d, e, l, up, max_depth,  background_color, background_light, pls, objects)
    
    # realiza escrita da imagem de saída
    with open('output.ppm', 'wb') as f:
        f.write(bytearray(f'P6 {h_res} {v_res} 255\n', 'ascii'))
        byteimg = array.array('B', list(pic.flatten()))
        byteimg.tofile(f)