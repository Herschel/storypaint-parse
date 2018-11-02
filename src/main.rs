extern crate byteorder;
use byteorder::{ReadBytesExt, LittleEndian};

extern crate clap;
extern crate hound;
extern crate image;
use image::{Pixel, Rgb};

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};

type Index = HashMap<String, IndexEntry>;

#[derive(Debug,PartialEq,Clone)]
pub struct IndexEntry {
  pub name: String,
  pub offset: u32,
  pub size: u32,
}

type Palette = [Rgb<u8>; 256];

#[derive(Clone)]
pub struct Background {
	pub width: u16,
	pub height: u16,
	pub palette: Palette,
}

pub struct BitReader<R: Read> {
	inner: R,
	cur_byte: u8,
	bit_index: u8,
	pos: usize,
}

impl<R: Read> BitReader<R> {
	fn new(inner: R) -> Self {
		BitReader {
			inner,
			cur_byte: 0,
			bit_index: 8,
			pos: 0,
		}
	}

	fn read_bit(&mut self) -> std::io::Result<bool> {
		if self.bit_index == 8 {
			self.cur_byte = self.inner.read_u8()?;
			self.bit_index = 0;
			self.pos += 1;
		}

		let val = self.cur_byte & (1 << self.bit_index);
		self.bit_index += 1;
		Ok(val != 0)
	}

	fn read_bits(&mut self, num_bits: u32) -> std::io::Result<u32> {
		let mut val = 0u32;
		for i in 0..num_bits {
			val |= (self.read_bit()? as u32) << i;
		}
		Ok(val)
	}
}

fn main() {
    use clap::{Arg, App};
    let matches = App::new("Storypaint Resource Extractor")
        .version("1.0")
        .author("Mike Welsh")
        .about("Extracts resources from .RSC files used in Novotrade Story Painting adventures")
        .arg(Arg::with_name("INPUT")
            .help("Sets the input .RSC file")
            .required(true))
        .arg(Arg::with_name("resource")
            .short("r")
            .long("resource")
            .value_name("RESOURCE_NAME")
            .help("Sets a specific resource to extract")
            .takes_value(true))
        .arg(Arg::with_name("OUTPUT")
            .help("Sets the output path for extracted asset(s)")
            .required(true))
        .get_matches();

    let input_filename = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

    match extract_rsc_file(&input_filename, &output_path, matches.value_of("resource")) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(-1);
        }
    }
}

fn extract_rsc_file(input_filename: &str, output_path: &str, resource: Option<&str>) -> Result<(), Box<Error>> {
    // Read input file.
    let data = {
        let mut data = vec![];
        let file = File::open(input_filename)?;
        let mut buf_reader = BufReader::new(file);
        buf_reader.read_to_end(&mut data)?;
        data
    };

    // Parse the index.
    let index = read_index(&data[..])?;

    let mut resource_reader = Cursor::new(&data);
    if let Some(resource_name) = resource {
        // Extract specific resource.
        let entry = match index.get(resource_name) {
            Some(entry) => entry,
            None => return Err(format!("Resource {} not found", resource_name).into()),
        };

        println!("Extracting {}...", entry.name);

        let resource_data = read_resource(&mut resource_reader, entry)?;

        use std::ffi::OsStr;
        let path = std::path::Path::new(resource_name);
        match path.extension().and_then(OsStr::to_str) {
            Some("VOC") => write_voc(&resource_data[..], output_path)?,
            _ => {
                std::fs::create_dir_all(output_path)?;
                let out_file = format!("{}/{}", output_path, entry.name);

                let file = File::create(out_file)?;
                let mut writer = BufWriter::new(file);
                writer.write_all(&resource_data)?;
            }
        }
    } else {
        // Extract all resources.
        for entry in index.values() {
            println!("Extracting {}...", entry.name);

            let resource_data = read_resource(&mut resource_reader, entry)?;

            std::fs::create_dir_all(output_path)?;
            let out_file = format!("{}/{}", output_path, entry.name);

            let file = File::create(out_file)?;
            let mut writer = BufWriter::new(file);
            writer.write_all(&resource_data)?;
        }
    }

    Ok(())
}

fn read_index<R: Read>(mut reader: R) -> Result<Index, Box<Error>> {
    let index_len = reader.read_u16::<LittleEndian>()?;
    let _length = reader.read_u16::<LittleEndian>()?;
    let _unknown = reader.read_u32::<LittleEndian>()?;

    if index_len % 0x17 != 0 {
        return Err("Invalid index size {}, must be multiple of 23".into());
    }
    let num_entries = index_len / 0x17;

    let mut index = HashMap::new();
    for _ in 0..num_entries {
        let entry = read_index_entry(&mut reader)?;
        index.insert(entry.name.clone(), entry);
    }

    Ok(index)
}

fn read_index_entry<R: Read>(mut reader: R) -> Result<IndexEntry, Box<Error>> {
    let mut name = String::new();
    let mut name_raw = [0u8; 13];
    reader.read_exact(&mut name_raw)?;
    for c in name_raw.into_iter() {
        if *c == 0 {
            break;
        }
        name.push(*c as char);
    }

    let _flags = reader.read_u16::<LittleEndian>()?;
    let size = reader.read_u32::<LittleEndian>()?;
    let offset = reader.read_u32::<LittleEndian>()?;

    Ok(IndexEntry {
        name,
        offset,
        size
    })
}

fn read_resource<R: Read + Seek>(mut reader: R, entry: &IndexEntry) -> Result<Vec<u8>, Box<Error>> {
    reader.seek(SeekFrom::Start(entry.offset.into()))?;

    let is_compressed = reader.read_u8()? != 0;
    let uncompressed_len = reader.read_u32::<LittleEndian>()?;
    let compressed_len = reader.read_u32::<LittleEndian>()?;

    let mut resource_reader = reader.take(compressed_len.into());

    let data = if is_compressed {
        decompress(resource_reader, uncompressed_len)?
    } else {
        if compressed_len != uncompressed_len {
            return Err("Uncompressed resource with different compressed length".into())
        }

        let mut data = vec![];
        resource_reader.read_to_end(&mut data)?;
        data
    };

    
    Ok(data)
}

fn read_palette<R: Read>(mut reader: R) -> Result<Palette, Box<Error>> {
	let mut palette = [Rgb::from_channels(0, 0, 0, 0); 256];
	for i in 0..256 {
		palette[i] = Rgb::from_channels(reader.read_u8()?, reader.read_u8()?, reader.read_u8()?, 0);
	}
	Ok(palette)
}

fn read_background<R: Read>(mut reader: R) -> Result<Background, Box<Error>> {
	let width = reader.read_u16::<LittleEndian>()?;
	let height = reader.read_u16::<LittleEndian>()?;
	let image_len = reader.read_u16::<LittleEndian>()? * 64;
	std::io::copy(&mut reader.by_ref().take(16), &mut std::io::sink());

	let mut data = Vec::with_capacity(image_len as usize);
	reader.take(image_len as u64).read_to_end(&mut data)?;

	
	Err("a".into())
}

fn decompress<R: Read>(reader: R, uncompressed_len: u32) -> Result<Vec<u8>, Box<Error>> {
	let mut out: Vec<u8> = Vec::with_capacity(uncompressed_len as usize);
    let mut reader = BitReader::new(reader);

    let uncompressed_len: usize = uncompressed_len as usize;
    while out.len() < uncompressed_len {
        let is_byte = reader.read_bit()?;
        if is_byte {
            let byte = reader.read_bits(8)? as u8;
            out.push(byte);
        } else {
            let offset_bits = if reader.read_bit()? { 13 } else { 10 };
            let offset = reader.read_bits(offset_bits)?;
            let mut len_bits = 2;
            while !reader.read_bit()? {
                len_bits += 1;
            }
            let mask = (1 << len_bits) - 1;
            
            let len = reader.read_bits(len_bits)? + mask - 1;
            let src_pos = out.len() - 1 - offset as usize;
            for i in 0..len as usize {
                let val = out[src_pos + i];
                out.push(val);
            }
        }
    }

	Ok(out)
}

fn write_voc<R: Read>(reader: R, out_file: &str) -> Result<(), Box<Error>> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    let spec = WavSpec {
        channels: 1,
        sample_rate: 11025,
        bits_per_sample: 8,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(out_file, spec)?;
    for sample in reader.bytes() {
        let sample = (i16::from(sample?) - 128) as i8;
        writer.write_sample(sample)?;
    }

    Ok(())
}