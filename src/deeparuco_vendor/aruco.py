from random import randint, random

import cv2
import numpy as np

MIP36h12 = [
    0xD2B63A09D,
    0x6001134E5,
    0x1206FBE72,
    0xFF8AD6CB4,
    0x85DA9BC49,
    0xB461AFE9C,
    0x6DB51FE13,
    0x5248C541F,
    0x8F34503,
    0x8EA462ECE,
    0xEAC2BE76D,
    0x1AF615C44,
    0xB48A49F27,
    0x2E4E1283B,
    0x78B1F2FA8,
    0x27D34F57E,
    0x89222FFF1,
    0x4C1669406,
    0xBF49B3511,
    0xDC191CD5D,
    0x11D7C3F85,
    0x16A130E35,
    0xE29F27EFF,
    0x428D8AE0C,
    0x90D548477,
    0x2319CBC93,
    0xC3B0C3DFC,
    0x424BCCC9,
    0x2A081D630,
    0x762743D96,
    0xD0645BF19,
    0xF38D7FD60,
    0xC6CBF9A10,
    0x3C1BE7C65,
    0x276F75E63,
    0x4490A3F63,
    0xDA60ACD52,
    0x3CC68DF59,
    0xAB46F9DAE,
    0x88D533D78,
    0xB6D62EC21,
    0xB3C02B646,
    0x22E56D408,
    0xAC5F5770A,
    0xAAA993F66,
    0x4CAA07C8D,
    0x5C9B4F7B0,
    0xAA9EF0E05,
    0x705C5750,
    0xAC81F545E,
    0x735B91E74,
    0x8CC35CEE4,
    0xE44694D04,
    0xB5E121DE0,
    0x261017D0F,
    0xF1D439EB5,
    0xA1A33AC96,
    0x174C62C02,
    0x1EE27F716,
    0x8B1C5ECE9,
    0x6A05B0C6A,
    0xD0568DFC,
    0x192D25E5F,
    0x1ADBECCC8,
    0xCFEC87F00,
    0xD0B9DDE7A,
    0x88DCEF81E,
    0x445681CB9,
    0xDBB2FFC83,
    0xA48D96DF1,
    0xB72CC2E7D,
    0xC295B53F,
    0xF49832704,
    0x9968EDC29,
    0x9E4E1AF85,
    0x8683E2D1B,
    0x810B45C04,
    0x6AC44BFE2,
    0x645346615,
    0x3990BD598,
    0x1C9ED0F6A,
    0xC26729D65,
    0x83993F795,
    0x3AC05AC5D,
    0x357ADFF3B,
    0xD5C05565,
    0x2F547EF44,
    0x86C115041,
    0x640FD9E5F,
    0xCE08BBCF7,
    0x109BB343E,
    0xC21435C92,
    0x35B4DFCE4,
    0x459752CF2,
    0xEC915B82C,
    0x51881EED0,
    0x2DDA7DC97,
    0x2E0142144,
    0x42E890F99,
    0x9A8856527,
    0x8E80D9D80,
    0x891CBCF34,
    0x25DD82410,
    0x239551D34,
    0x8FE8F0C70,
    0x94106A970,
    0x82609B40C,
    0xFC9CAF36,
    0x688181D11,
    0x718613C08,
    0xF1AB7629,
    0xA357BFC18,
    0x4C03B7A46,
    0x204DEDCE6,
    0xAD6300D37,
    0x84CC4CD09,
    0x42160E5C4,
    0x87D2ADFA8,
    0x7850E7749,
    0x4E750FC7C,
    0xBF2E5DFDA,
    0xD88324DA5,
    0x234B52F80,
    0x378204514,
    0xABDF2AD53,
    0x365E78EF9,
    0x49CAA6CA2,
    0x3C39DDF3,
    0xC68C5385D,
    0x5BFCBBF67,
    0x623241E21,
    0xABC90D5CC,
    0x388C6FE85,
    0xDA0E2D62D,
    0x10855DFE9,
    0x4D46EFD6B,
    0x76EA12D61,
    0x9DB377D3D,
    0xEED0EFA71,
    0xE6EC3AE2F,
    0x441FAEE83,
    0xBA19C8FF5,
    0x313035EAB,
    0x6CE8F7625,
    0x880DAB58D,
    0x8D3409E0D,
    0x2BE92EE21,
    0xD60302C6C,
    0x469FFC724,
    0x87EEBEED3,
    0x42587EF7A,
    0x7A8CC4E52,
    0x76A437650,
    0x999E41EF4,
    0x7D0969E42,
    0xC02BAF46B,
    0x9259F3E47,
    0x2116A1DC0,
    0x9F2DE4D84,
    0xEFFAC29,
    0x7B371FF8C,
    0x668339DA9,
    0xD010AEE3F,
    0x1CD00B4C0,
    0x95070FC3B,
    0xF84C9A770,
    0x38F863D76,
    0x3646FF045,
    0xCE1B96412,
    0x7A5D45DA8,
    0x14E00EF6C,
    0x5E95ABFD8,
    0xB2E9CB729,
    0x36C47DD7,
    0xB8EE97C6B,
    0xE9E8F657,
    0xD4AD2EF1A,
    0x8811C7F32,
    0x47BDE7C31,
    0x3ADADFB64,
    0x6E5B28574,
    0x33E67CD91,
    0x2AB9FDD2D,
    0x8AFA67F2B,
    0xE6A28FC5E,
    0x72049CDBD,
    0xAE65DAC12,
    0x1251A4526,
    0x1089AB841,
    0xE2F096EE0,
    0xB0CAEE573,
    0xFD6677E86,
    0x444B3F518,
    0xBE8B3A56A,
    0x680A75CFC,
    0xAC02BAEA8,
    0x97D815E1C,
    0x1D4386E08,
    0x1A14F5B0E,
    0xE658A8D81,
    0xA3868EFA7,
    0x3668A9673,
    0xE8FC53D85,
    0x2E2B7EDD5,
    0x8B2470F13,
    0xF69795F32,
    0x4589FFC8E,
    0x2E2080C9C,
    0x64265F7D,
    0x3D714DD10,
    0x1692C6EF1,
    0x3E67F2F49,
    0x5041DAD63,
    0x1A1503415,
    0x64C18C742,
    0xA72EEC35,
    0x1F0F9DC60,
    0xA9559BC67,
    0xF32911D0D,
    0x21C0D4FFC,
    0xE01CEF5B0,
    0x4E23A3520,
    0xAA4F04E49,
    0xE1C4FCC43,
    0x208E8F6E8,
    0x8486774A5,
    0x9E98C7558,
    0x2C59FB7DC,
    0x9446A4613,
    0x8292DCC2E,
    0x4D61631,
    0xD05527809,
    0xA0163852D,
    0x8F657F639,
    0xCCA6C3E37,
    0xCB136BC7A,
    0xFC5A83E53,
    0x9AA44FC30,
    0xBDEC1BD3C,
    0xE020B9F7C,
    0x4B8F35FB0,
    0xB8165F637,
    0x33DC88D69,
    0x10A2F7E4D,
    0xC8CB5FF53,
    0xDE259FF6B,
    0x46D070DD4,
    0x32D3B9741,
    0x7075F1C04,
    0x4D58DBEA0,
    0x000000000,
    0xFFFFFFFFF,
]


def id_to_bits(id):
    return [float(bit) for bit in format(MIP36h12[id], "036b")]


def get_marker(id, size=512, border_width=1.0):

    marker = np.zeros((8, 8, 4), dtype=np.uint8)
    marker[0:8, 0:8] = (0, 0, 0, 255.0)
    marker[1:7, 1:7] = cv2.cvtColor(
        np.reshape(
            np.asarray([(i * 255.0) for i in id_to_bits(id)]).astype(np.uint8), (6, 6)
        ),
        cv2.COLOR_GRAY2BGRA,
    )

    canvas = np.ones((size, size, 4), dtype=np.uint8) * 255.0
    canvas[:, :, 3] = 0

    center = size // 2
    bg_size = int(size - 2 * (size / 10) * (1 - border_width))
    canvas[
        center - bg_size // 2 : center + bg_size // 2,
        center - bg_size // 2 : center + bg_size // 2,
        3,
    ] = 255.0

    wo_border = int(size - 2 * (size / 10))
    canvas[
        (size - wo_border) // 2 : (size + wo_border) // 2,
        (size - wo_border) // 2 : (size + wo_border) // 2,
    ] = cv2.resize(marker, (wo_border, wo_border), interpolation=cv2.INTER_NEAREST)

    corners = [
        [(size - wo_border) // 2, (size - wo_border) // 2],
        [(size - wo_border) // 2, (size + wo_border) // 2 - 1],
        [(size + wo_border) // 2 - 1, (size + wo_border) // 2 - 1],
        [(size + wo_border) // 2 - 1, (size - wo_border) // 2],
    ]

    return canvas, corners


ids_as_bits = [id_to_bits(i) for i in range(250)]


def find_id(bits):

    rot0 = bits.flatten()
    rot90 = np.rot90(bits, 1).flatten()
    rot180 = np.rot90(bits, 2).flatten()
    rot270 = np.rot90(bits, 3).flatten()

    distances = [
        int(
            np.min(
                [
                    np.sum(np.abs(rot0 - check_bits)),
                    np.sum(np.abs(rot90 - check_bits)),
                    np.sum(np.abs(rot180 - check_bits)),
                    np.sum(np.abs(rot270 - check_bits)),
                ]
            )
        )
        for check_bits in ids_as_bits
    ]

    id = int(np.argmin(distances))

    return (id, distances[id])


if __name__ == "__main__":
    marker, corners = get_marker(randint(0, 250), border_width=random())
    for c in corners:
        cv2.circle(marker, (c[0], c[1]), 5, (0, 255, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite("test_aruco.png", marker)
